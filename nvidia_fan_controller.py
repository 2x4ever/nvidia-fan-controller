#!/usr/bin/env python3

import re
import os
import sys
import logging
from typing import Dict, List, Optional, Tuple
from contextlib import AbstractContextManager
import nvidia_smi
from pynvml import (
    nvmlDeviceGetSupportedMemoryClocks,
    nvmlDeviceGetMaxClockInfo,
    _nvmlGetFunctionPointer,
    _nvmlCheckReturn,
    nvmlDeviceGetTemperature,
    NVML_TEMPERATURE_GPU,
)
from ctypes import *
from ctypes.util import find_library

NV_DEVICE_COUNT = 0
NV_DEVICE_HANDLES = []
NV_DEVICE_FAN_COUNT = []

def nv_device_count():
    return nvidia_smi.nvmlDeviceGetCount()

def nv_device_temperature(idx):
    handle = nvidia_smi.nvmlDeviceGetHandleByIndex(i)
    temp = int(nvmlDeviceGetTemperature(handle, NVML_TEMPERATURE_GPU))
    temp *= 1000
    return temp

def nvmlDeviceGetFanSpeed_v2(handle, fan):
    c_speed = c_uint()
    c_fan = c_uint(fan)
    fn = _nvmlGetFunctionPointer("nvmlDeviceGetFanSpeed_v2")
    ret = fn(handle, c_fan, byref(c_speed))
    _nvmlCheckReturn(ret)
    return c_speed.value

def nvmlDeviceGetNumFans(handle):
    c_num_fans = c_uint()
    fn = _nvmlGetFunctionPointer("nvmlDeviceGetNumFans")
    ret = fn(handle, byref(c_num_fans))
    _nvmlCheckReturn(ret)
    return c_num_fans.value

def nvmlDeviceGetMinMaxFanSpeed(handle):
    c_min_speed = c_uint()
    c_max_speed = c_uint()
    fn = _nvmlGetFunctionPointer("nvmlDeviceGetMinMaxFanSpeed")
    ret = fn(handle, byref(c_min_speed), byref(c_max_speed))
    _nvmlCheckReturn(ret)
    return (c_min_speed.value, c_max_speed.value)

def nvmlDeviceSetFanSpeed_v2(handle, fan, speed):
    c_speed = c_uint(speed)
    c_fan = c_uint(fan)
    fn = _nvmlGetFunctionPointer("nvmlDeviceSetFanSpeed_v2")
    ret = fn(handle, c_fan, c_speed)
    _nvmlCheckReturn(ret)
    return ret

NVML_FAN_POLICY_TEMPERATURE_CONTINOUS_SW = 0
NVML_FAN_POLICY_MANUAL                   = 1

def nvmlDeviceSetFanControlPolicy(handle, fan, policy):
    c_policy = c_uint(policy)
    c_fan = c_uint(fan)
    fn = _nvmlGetFunctionPointer("nvmlDeviceSetFanControlPolicy")
    ret = fn(handle, c_fan, c_policy)
    _nvmlCheckReturn(ret)
    return ret


logger = logging.getLogger('nvidia-fan-controller')


SERVICE_FILE_TEMPLATE = """
[Unit]
Description=Nvidia GPU Fan Controller

[Service]
Type=simple
User={USER}
Group={USER}
PIDFile=/run/nvidia-fan-control.pid
ExecStart=/usr/bin/python3 {FILEPATH} --target-temperature {TARGET_TEMPERATURE} --interval-secs {INTERVAL_SECS} --log-level INFO
Restart=always
RestartSec=10

[Install]
WantedBy=multi-user.target

"""


class PIDController:
    """
    PID Controller using variable names defined in:

        https://en.wikipedia.org/wiki/PID_controller

    Args:
        x_target: setpoint for process variable :math:`x`
        u_min: minimal value of control variable :math:`u`
        u_max: maximal value of control variable :math:`u`
        u_start: initial value of control variable :math:`u`
        reverse: whether to flip the sign of the residuals (for reverse control task)
        Kp: coefficient of proportionality term
        Ki: coefficient of integral term
        Kd: coefficient of derivative term
        gamma: discount factor for exponentially annealing past contributions to integral term
        alpha: smoothing parameter for computing a rolling average error
        e_min: lower bound for the error = x_observed - x_target
        e_max: upper bound for the error = x_observed - x_target
        e_total_min: lower bound for the accumulated error
        e_total_max: upper bound for the accumulated error

    """
    def __init__(
            self,
            x_target: float,
            u_min: float,
            u_max: float,
            u_start: Optional[float] = None,
            reverse: bool = False,
            Kp: float = 0.5,
            Ki: float = 1.0,
            Kd: float = 1.0,
            gamma: float = 0.9,
            alpha: float = 0.5,
            e_min: float = -float('inf'),
            e_max: float = float('inf'),
            e_total_min: float = -float('inf'),
            e_total_max: float = float('inf')):

        self.u_min = u_min
        self.u_max = u_max
        self.u_start = (u_min + u_max) / 2 if u_start is None else u_start
        self.x_target = x_target
        self.reverse = reverse
        self.Kp = Kp
        self.Ki = Ki
        self.Kd = Kd
        self.gamma = gamma
        self.alpha = alpha
        self.e_min = e_min
        self.e_max = e_max
        self.e_total_min = e_total_min
        self.e_total_max = e_total_max
        self.reset_state()

    @property
    def x_target(self) -> float:
        return self.__x_target

    @x_target.setter
    def x_target(self, x_target_new: float) -> None:
        self.__x_target = x_target_new
        self.reset_state()

    def reset_state(self) -> None:
        self._u = self.u_start
        self._e_total = 0.
        self._e_prev = 0.

    def __repr__(self) -> str:
        config = ', '.join(f'{k}={v:r}' for k, v in self.__dict__.items() if not k.startswith('_'))
        state = ', '.join(f'{k}: {v:r}' for k, v in self.get_state().items())
        return f'{type(self).__name__}({config}) {{ {state} }}'

    def get_state(self) -> Dict[str, float]:
        """Return a summary of the current state."""
        return {
            'u': self._u,
            'e_prev': self._e_prev,
            'e_total': self._e_total,
            'x_target': self.x_target,
        }

    def __call__(self, x_obs: float) -> float:
        """
        Update internal state of the controller and return new control variable :math:`u`.

        Args:
            x_obs: observation of the process variable :math:`x`

        Returns:
            u: new value for the control variable :math:`u`

        """
        e = x_obs - self.x_target

        # Clip residual.
        e = max(self.e_min, min(self.e_max, e))

        if self.reverse:
            e = -e

        # use Polyak smoothing for the proportional term
        e_smooth = self.alpha * e + (1 - self.alpha) * self._e_prev

        # PID terms (proportional, integral, derivative)
        p = e_smooth
        i = e + self.gamma * self._e_total
        d = e - self._e_prev

        # control variable (clipped to provided range)
        u = max(self.u_min, min(self.u_max, self.Kp * p + self.Ki * i + self.Kd * d))

        logger.debug(  # pylint: disable=logging-fstring-interpolation
            "Kp * p + Ki * i + Kd * d = "
            f"{self.Kp} * {p:.2f} + {self.Ki} * {i:.2f} + {self.Kd} * {d:.2f} = "
            f"{self.Kp * p:.2f} + {self.Ki * i:.2f} + {self.Kd * d:.2f} = "
            f"{self.Kp * p + self.Ki * i + self.Kd * d:.2f}, "
            f"u_clipped = {u:.2f}")

        # update state
        if u != self._u:
            self._u = u
            self._e_total = max(self.e_total_min, min(self.e_total_max, i))
            self._e_prev = e_smooth

        return self._u


class ManualFanControl(AbstractContextManager):
    """ Context manager that sets manual fan control only for the duration of the context. """
    def __enter__(self):
        logger.debug("enabling manual gpu fan control")
        nv_set_all_fans_manual_mode()

    def __exit__(self, exc_type, exc_value, traceback):
        logger.debug("disabling manual gpu fan control")
        nv_set_all_fans_auto_mode()


def nv_set_all_fans_manual_mode():
    for idx in range(NV_DEVICE_COUNT):
        for fan in range(NV_DEVICE_FAN_COUNT[idx]):
            nvmlDeviceSetFanControlPolicy(NV_DEVICE_HANDLES[idx], fan, NVML_FAN_POLICY_MANUAL)

def nv_set_all_fans_auto_mode():
    for idx in range(NV_DEVICE_COUNT):
        for fan in range(NV_DEVICE_FAN_COUNT[idx]):
            nvmlDeviceSetFanControlPolicy(NV_DEVICE_HANDLES[idx], fan, NVML_FAN_POLICY_TEMPERATURE_CONTINOUS_SW)

def run_cmd(cmd: List[str]) -> str:
    logger.debug("Running cmd: %s", ' '.join(cmd))
    p = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    p.wait()

    if p.returncode:
        logger.critical("Unable to run cmd: %s", ' '.join(cmd))
        if p.stderr is not None:
            for line_bytes in p.stderr.readlines():
                line = line_bytes.decode().strip()
                if line:
                    logger.error("Caught process stderr: %s", line)
            raise subprocess.CalledProcessError(p.returncode, cmd)

    if p.stdout is None:
        return ''

    return p.stdout.read().decode()


def get_measurements() -> List[Tuple[int, int, int]]:
#    stdout = run_cmd(['nvidia-smi', '--query-gpu=index,temperature.gpu,fan.speed', '--format=csv,noheader'])
#    measurements = re.findall(r'(\d+), (\d+), (\d+) %', stdout, flags=re.MULTILINE)
#    measurements = [tuple(map(int, values)) for values in measurements]  # parse ints
    measurements = list()
    for idx in range(NV_DEVICE_COUNT):
        temperature = nvmlDeviceGetTemperature(NV_DEVICE_HANDLES[idx], NVML_TEMPERATURE_GPU)
        fan_speed = nvmlDeviceGetFanSpeed_v2(NV_DEVICE_HANDLES[idx], 0)
        measurements.append((idx, temperature, fan_speed))
    print(measurements)
    return measurements  # [(index, temperature, fanspeed)]


def get_fan_speed(index: int) -> int:
    fan_speed = nvmlDeviceGetFanSpeed_v2(NV_DEVICE_HANDLES[index], 0)
    logger.debug("Current fan speed setting: [fan-%d]/GPUTargetFanSpeed=%s", index, fan_speed)
    return int(fan_speed)


def set_fan_speed(index: int, fan_speed: int) -> None:
    logger.info("Setting new fan[%d] speed: %s", index, fan_speed)
    for fan in range(NV_DEVICE_FAN_COUNT[index]):
        nvmlDeviceSetFanSpeed_v2(NV_DEVICE_HANDLES[index], fan, fan_speed)


def create_service_file(target_temperature: int = 60, interval_secs: int = 2) -> None:
    script_filepath = os.path.abspath(__file__)
    service_filepath = os.path.join(os.path.dirname(script_filepath), 'nvidia-fan-controller.service')

    content = SERVICE_FILE_TEMPLATE.format(
        FILEPATH=script_filepath, TARGET_TEMPERATURE=target_temperature, INTERVAL_SECS=interval_secs, **os.environ)

    logger.info("Creating/replacing service file at: %s", service_filepath)
    logger.debug("Creating/replacing service file content:\n%r", content)
    with open(service_filepath, 'w', encoding='utf-8') as f:
        f.write(content)

def nvml_init():
    global NV_DEVICE_COUNT
    global NV_DEVICE_HANDLES
    nvidia_smi.nvmlInit()
    NV_DEVICE_COUNT = nv_device_count()
    for idx in range(NV_DEVICE_COUNT):
        NV_DEVICE_HANDLES.append(nvidia_smi.nvmlDeviceGetHandleByIndex(idx))
        NV_DEVICE_FAN_COUNT.append(nvmlDeviceGetNumFans(NV_DEVICE_HANDLES[idx]))


def main() -> None:
    import argparse
    from time import sleep

    parser = argparse.ArgumentParser()
    parser.add_argument('--target-temperature', type=int, default=60, help="target max temperature (Celcius)")
    parser.add_argument('--interval-secs', type=int, default=2, help="number of seconds between consecutive updates")
    parser.add_argument('--log-level', choices=('DEBUG', 'INFO', 'WARN'), default='INFO', help="verbosity level")
    parser.add_argument('--create-service-file', action='store_true', help="create service file and exit")
    args = parser.parse_args()

    logging.basicConfig(level=getattr(logging, args.log_level))

    if args.create_service_file:
        create_service_file(target_temperature=args.target_temperature, interval_secs=args.interval_secs)
        sys.exit()

    nvml_init()

    if not get_measurements():
        raise RuntimeError("no gpu detected")

    # give each GPU its own controller
    controllers = {}
    for index, temp, speed in get_measurements():
        (min_speed, max_speed) = nvmlDeviceGetMinMaxFanSpeed(NV_DEVICE_HANDLES[index])
        controller = PIDController(x_target=args.target_temperature, u_min=min_speed, u_max=max_speed, u_start=max(temp / 0.9, speed), e_total_min=-10)
        controllers[index] = controller

    with ManualFanControl():
        while True:
            for index, temp, _ in get_measurements():
                # new speed proposed by PID-controller
                controller = controllers[index]
                print(f"{args.target_temperature} - {temp}")
                fan_speed = int(round(controller(temp)))

                # only update if change is non-trivial
                if fan_speed != get_fan_speed(index):
                    set_fan_speed(index, fan_speed)

            sleep(args.interval_secs)


if __name__ == '__main__':
    main()

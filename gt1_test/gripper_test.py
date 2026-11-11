"""gripper_test: accionar cada acción del gripper sin mover el robot.

Sirve para comprobar el cableado y el archivo de un gripper nuevo: cada
acción de ``config/grippers/<gripper>.toml`` queda en una tecla (1, 2, …) y
la pantalla muestra el estado de las salidas que usa. Al iniciar, el TCP y la
carga del gripper se cargan en el robot y se verifican.

Teclas:
    1..9   ejecutar la acción correspondiente (la lista está en pantalla)
"""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass

from gt1.config.schemas import DigitalOutputGripperConfig, StrictModel
from gt1.domain import DigitalOutput
from gt1.hardware.gripper import Gripper
from gt1.hardware.robot import Robot
from gt1.runtime.console import OperatorConsole
from gt1.runtime.factories import build_gripper, build_robot
from gt1.runtime.overlay import text_panel
from gt1.runtime.program import Program, run_program
from gt1.runtime.run_log import RunLog
from gt1.runtime.runner import Job
from gt1.runtime.station import StationHardware

PROGRAM = "gripper_test"
PANEL_SIZE_PX = (900, 420)


class Components(StrictModel):
    robot: str
    gripper: str


class GripperTestConfig(StrictModel):
    components: Components


@dataclass
class GripperTest:
    robot: Robot
    gripper: Gripper
    outputs: tuple[DigitalOutput, ...]
    console: OperatorConsole
    run_log: RunLog

    def job(self, action: str) -> Job:
        def run(stop: threading.Event) -> None:
            started = time.perf_counter()
            self.gripper.perform(action)
            duration_s = time.perf_counter() - started
            states = self._output_states()
            self.run_log.record(
                {
                    "action": action,
                    "duration_s": round(duration_s, 3),
                    "outputs": {output.describe(): value for output, value in states.items()},
                },
                {},
            )
            self.show_panel(f"Última acción: {action} ({duration_s:.2f} s)")
            self.console.set_status(f"{action}: listo en {duration_s:.2f} s")

        return run

    def show_panel(self, last_event: str) -> None:
        lines = [f"Gripper: {self.gripper.name}", ""]
        lines += [
            f"  {index}  {action.name}: {action.description}"
            for index, action in enumerate(self.gripper.actions, start=1)
        ]
        lines += [
            "",
            "Salidas: "
            + "   ".join(
                f"{output.describe()}={int(value)}" for output, value in self._output_states().items()
            ),
            "",
            last_event,
        ]
        self.console.show(text_panel(lines, PANEL_SIZE_PX))

    def _output_states(self) -> dict[DigitalOutput, bool]:
        return {output: self.robot.digital_output(output) for output in self.outputs}


def gripper_test(program: Program[GripperTestConfig]) -> None:
    components = program.config.components
    repository = program.repository
    gripper_config = repository.gripper(components.gripper)
    if not isinstance(gripper_config, DigitalOutputGripperConfig):
        raise ValueError(f"El gripper {components.gripper} no tiene acciones que probar.")
    outputs = tuple(
        dict.fromkeys(
            setting.digital_output()
            for action in gripper_config.actions.values()
            for setting in action.outputs
        )
    )

    with build_robot(repository.robot(components.robot)) as robot:
        gripper = build_gripper(components.gripper, gripper_config, robot)
        program.confirm_station(
            StationHardware(
                camera=None,
                camera_name=None,
                robot=robot,
                robot_name=components.robot,
                gripper=gripper,
                tray_name=None,
            )
        )
        console = OperatorConsole(f"GT1 - {PROGRAM}")
        test = GripperTest(robot, gripper, outputs, console, program.run_log)
        for index, action in enumerate(gripper.actions[:9], start=1):
            console.bind_job(str(index), f"{action.name}: {action.description}", test.job(action.name))
        test.show_panel("Listo.")
        console.run()


def main() -> None:
    run_program("tests", PROGRAM, GripperTestConfig, gripper_test)


if __name__ == "__main__":
    main()

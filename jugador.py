from random import randint, uniform
import random
from abc import ABC, abstractmethod
from utils import puntaje_y_no_usados, JUGADA_PLANTARSE, JUGADA_TIRAR
import os
import csv


class Jugador(ABC):
    @abstractmethod
    def jugar(
        self,
        puntaje_total: int,
        puntaje_turno: int,
        dados: list[int],
        verbose: bool = False,
    ) -> tuple[int, list[int]]:
        pass


class JugadorAleatorio(Jugador):
    def __init__(self, nombre: str):
        self.nombre = nombre

    def jugar(
        self,
        puntaje_total: int,
        puntaje_turno: int,
        dados: list[int],
        verbose: bool = False,
    ) -> tuple[int, list[int]]:
        (puntaje, no_usados) = puntaje_y_no_usados(dados)
        if randint(0, 1) == 0:
            return (JUGADA_PLANTARSE, [])
        else:
            return (JUGADA_TIRAR, no_usados)


class JugadorSiempreSePlanta(Jugador):
    def __init__(self, nombre: str):
        self.nombre = nombre

    def jugar(
        self,
        puntaje_total: int,
        puntaje_turno: int,
        dados: list[int],
        verbose: bool = False,
    ) -> tuple[int, list[int]]:
        return (JUGADA_PLANTARSE, [])


class ElBatoQueSoloCalculaPromedios(Jugador):
    def __init__(
        self, epsilon: float, politica_csv_path: str, is_training: bool = False
    ):
        self.nombre = "Monte Carlo"
        self.politica_csv_path = politica_csv_path
        self.epsilon = epsilon  # e-greedy
        self.history = []
        self.estados = {}
        self.is_training = is_training

        if self.is_training:
            # Elimina el archivo si ya existe
            if os.path.exists(self.politica_csv_path):
                os.remove(self.politica_csv_path)
            self._crear_csv()
        self._cargar_estados()

    def _crear_csv(self):
        estados_base = {
            0: {"tirar": 0, "plantarse": 0, "c_tirar": 1, "c_plantarse": 1},
            1: {"tirar": 0, "plantarse": 0, "c_tirar": 1, "c_plantarse": 1},
            2: {"tirar": 0, "plantarse": 0, "c_tirar": 1, "c_plantarse": 1},
            3: {"tirar": 0, "plantarse": 0, "c_tirar": 1, "c_plantarse": 1},
            4: {"tirar": 0, "plantarse": 0, "c_tirar": 1, "c_plantarse": 1},
            5: {"tirar": 0, "plantarse": 0, "c_tirar": 1, "c_plantarse": 1},
        }
        with open(self.politica_csv_path, mode="w", newline="") as file:
            writer = csv.writer(file)
            # Escribe el encabezado
            writer.writerow(["estado", "tirar", "plantarse", "c_tirar", "c_plantarse"])

            for estado, valores in estados_base.items():
                fila = [estado] + list(valores.values())
                writer.writerow(fila)

    def _cargar_estados(self):
        with open(self.politica_csv_path, mode="r") as file:
            reader = csv.DictReader(file)
            for row in reader:
                estado = int(row["estado"])
                self.estados[estado] = {
                    "tirar": int(row["tirar"]),
                    "plantarse": int(row["plantarse"]),
                    "c_tirar": int(row["c_tirar"]),
                    "c_plantarse": int(row["c_plantarse"]),
                }

    def guardar_estados_en_csv(self):
        """Guarda el contenido de self.estados en el archivo CSV."""
        with open(self.politica_csv_path, mode="w", newline="") as file:
            writer = csv.writer(file)
            # Escribe el encabezado
            writer.writerow(["estado", "tirar", "plantarse", "c_tirar", "c_plantarse"])

            for estado, valores in self.estados.items():
                fila = [
                    estado,
                    valores["tirar"],
                    valores["plantarse"],
                    valores["c_tirar"],
                    valores["c_plantarse"],
                ]
                writer.writerow(fila)

    def print_table(self):
        for state, rewards in self.estados.items():
            reward_plantarse = rewards["plantarse"]
            reward_tirar = rewards["tirar"]
            avg_reward_plantarse = reward_plantarse / rewards["c_plantarse"]
            avg_reward_tirar = reward_tirar / rewards["c_tirar"]
            ct = rewards["c_tirar"]
            cp = rewards["c_plantarse"]

            print(f"State {state}:")
            print(f"  Cantidad plantarse: {cp:.2f}")
            print(f"  Cantidad tirar: {ct:.2f}")
            print(f"  Promedio reward_plantarse: {avg_reward_plantarse:.2f}")
            print(f"  Promedio reward_tirar: {avg_reward_tirar:.2f}")

    def jugar(self, puntaje_total: int, puntaje_turno: int, dados: list[int]):
        (puntaje, no_usados) = puntaje_y_no_usados(dados)
        cant_dados = len(no_usados)
        if self.is_training and uniform(0, 1) < self.epsilon:
            if uniform(0, 1) > 0.5:
                self.history.append((cant_dados, "plantarse"))
                return (JUGADA_PLANTARSE, [])
            else:
                self.history.append((cant_dados, "tirar"))
                return (JUGADA_TIRAR, no_usados)
        else:
            if (
                self.estados[cant_dados]["tirar"] / self.estados[cant_dados]["c_tirar"]
                > self.estados[cant_dados]["plantarse"]
                / self.estados[cant_dados]["c_plantarse"]
            ):
                self.history.append((cant_dados, "tirar"))
                return (JUGADA_TIRAR, no_usados)
            elif (
                self.estados[cant_dados]["tirar"] / self.estados[cant_dados]["c_tirar"]
                < self.estados[cant_dados]["plantarse"]
                / self.estados[cant_dados]["c_plantarse"]
            ):
                self.history.append((cant_dados, "plantarse"))
                return (JUGADA_PLANTARSE, [])
            else:
                if uniform(0, 1) > 0.5:
                    return (JUGADA_TIRAR, no_usados)
                else:
                    return (JUGADA_PLANTARSE, [])

    def actualizar_tabla(self, estado, puntaje_turno):
        for estado, accion in self.history:
            self.estados[estado][accion] += puntaje_turno
            self.estados[estado]["c_" + accion] += 1
        self.history.clear()


class AgenteQLearning(Jugador):
    def __init__(self, alpha: float, gamma: float, epsilon: float):
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.estados = {
            0: {"tirar": 0, "plantarse": 0},
            1: {"tirar": 0, "plantarse": 0},
            2: {"tirar": 0, "plantarse": 0},
            3: {"tirar": 0, "plantarse": 0},
            4: {"tirar": 0, "plantarse": 0},
            5: {"tirar": 0, "plantarse": 0},
            6: {"tirar": 0, "plantarse": 0},
        }
        self.prev_state = 6
        self.prev_action = JUGADA_TIRAR

    def jugar(self, dados: list[int]):
        (puntaje, no_usados) = puntaje_y_no_usados(dados)
        cant_dados = len(no_usados)
        # self.actualizar_tabla(puntaje, cant_dados)
        self.prev_state = cant_dados
        if uniform(0, 1) < self.epsilon:
            if uniform(0, 1) > 0.5:
                self.prev_action = JUGADA_PLANTARSE
                return (JUGADA_PLANTARSE, [])
            else:
                self.prev_action = JUGADA_TIRAR
                return (JUGADA_TIRAR, no_usados)
        else:
            if (
                self.estados[cant_dados]["tirar"]
                > self.estados[cant_dados]["plantarse"]
            ):
                self.prev_action = JUGADA_TIRAR
                return (JUGADA_TIRAR, no_usados)
            else:
                self.prev_action = JUGADA_PLANTARSE
                return (JUGADA_PLANTARSE, [])

    def actualizar_tabla(self, puntaje: int, cant_dados: int):
        accion = "plantarse" if self.prev_action == JUGADA_PLANTARSE else "tirar"
        print(accion, self.prev_state)
        self.estados[self.prev_state][accion] = self.estados[self.prev_state][
            accion
        ] + self.alpha * (
            puntaje
            + self.gamma
            * max(
                self.estados[cant_dados]["tirar"], self.estados[cant_dados]["plantarse"]
            )
            - self.estados[self.prev_state][accion]
        )

    def print_table(self):
        for state, rewards in self.estados.items():
            reward_plantarse = rewards["plantarse"]
            reward_tirar = rewards["tirar"]
            print(f"State {state}:")
            print(f"  Promedio reward_plantarse: {reward_tirar:.2f}")
            print(f"  Promedio reward_tirar: {reward_plantarse:.2f}")

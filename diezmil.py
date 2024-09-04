from random import randint
from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np
from utils import puntaje_y_no_usados, separar, JUGADA_PLANTARSE, JUGADA_TIRAR
from jugador import (
    Jugador,
    JugadorAleatorio,
    JugadorSiempreSePlanta,
    ElBatoQueSoloCalculaPromedios,
    AgenteQLearning,
)


class JuegoDiezMil:
    def __init__(self, jugador: Jugador):
        self.jugador: Jugador = jugador

    def jugar(self, verbose: bool = False, tope_turnos: int = 1000) -> tuple[int, int]:
        """Juega un juego de 10mil para un jugador, hasta terminar o hasta
        llegar a tope_turnos turnos. Devuelve la cantidad de turnos que
        necesitó y el puntaje final.
        """
        turno: int = 0
        puntaje_total: int = 0
        while puntaje_total < 10000 and turno < tope_turnos:
            # Nuevo turno
            turno += 1
            puntaje_turno: int = 0
            msg: str = "turno " + str(turno) + ":"

            # Un turno siempre empieza tirando los 6 dados.
            jugada: int = JUGADA_TIRAR
            dados_a_tirar: list[int] = [1, 2, 3, 4, 5, 6]
            fin_de_turno: bool = False

            while not fin_de_turno:
                # Tira los dados que correspondan y calcula su puntaje.
                dados: list[int] = [randint(1, 6) for _ in range(len(dados_a_tirar))]

                (puntaje_tirada, _) = puntaje_y_no_usados(dados)
                msg += " " + "".join(map(str, dados)) + " "

                if puntaje_tirada == 0:
                    # Mala suerte, no suma nada. Pierde el turno.
                    fin_de_turno = True
                    puntaje_turno = 0
                    if isinstance(self.jugador, ElBatoQueSoloCalculaPromedios):
                        self.jugador.actualizar_tabla(len(dados_a_tirar), puntaje_turno)

                else:
                    # Bien, suma puntos. Preguntamos al jugador qué quiere hacer.
                    if isinstance(self.jugador, AgenteQLearning):
                        jugada, dados_a_tirar = self.jugador.jugar(dados)
                    else:
                        jugada, dados_a_tirar = self.jugador.jugar(
                            puntaje_total, puntaje_turno, dados
                        )

                    if jugada == JUGADA_PLANTARSE:
                        msg += "P"
                        fin_de_turno = True
                        puntaje_turno += puntaje_tirada
                        if isinstance(self.jugador, ElBatoQueSoloCalculaPromedios):
                            self.jugador.actualizar_tabla(
                                len(dados_a_tirar), puntaje_turno
                            )

                    elif jugada == JUGADA_TIRAR:
                        dados_a_separar = separar(dados, dados_a_tirar)
                        assert len(dados_a_separar) + len(dados_a_tirar) == len(dados)
                        puntaje_tirada, dados_no_usados = puntaje_y_no_usados(
                            dados_a_separar
                        )
                        assert puntaje_tirada > 0 and len(dados_no_usados) == 0
                        puntaje_turno += puntaje_tirada
                        if len(dados_a_tirar) == 0:
                            dados_a_tirar = [1, 2, 3, 4, 5, 6]
                            if isinstance(self.jugador, AgenteQLearning):
                                self.jugador.prev_state = 6
                                self.jugador.prev_action = JUGADA_TIRAR
                        msg += "T(" + "".join(map(str, dados_a_tirar)) + ") "

            puntaje_total += puntaje_turno
            msg += (
                " --> " + str(puntaje_turno) + " puntos. TOTAL: " + str(puntaje_total)
            )
            if verbose:
                print(msg)
        return (turno, puntaje_total)


def main():
    agents = 10
    games = 10000
    play_amounts = []
    for _ in tqdm(range(agents)):
        player_amounts = []
        jugador = ElBatoQueSoloCalculaPromedios(
            0.01, "politica_montecarlo_mala.csv", False
        )
        # jugador = JugadorAleatorio("random")
        for _ in tqdm(range(games)):
            juego = JuegoDiezMil(jugador)
            (cantidad_turnos, puntaje_final) = juego.jugar(verbose=False)
            player_amounts.append(cantidad_turnos)
        # jugador.print_table()
        play_amounts.append(player_amounts)
        jugador.guardar_estados_en_csv()

    # Convertir a numpy array
    play_amounts = np.array(play_amounts)

    # Calcular el promedio de las ultimas mil tiradas
    average_play_amounts = np.mean(play_amounts, axis=0)
    average_total_ult_mil = np.mean(average_play_amounts[-1000:])

    plt.figure(figsize=(10, 6))
    plt.plot(
        range(len(average_play_amounts)),
        average_play_amounts,
        label="Average Play Amounts",
        color="red",
    )
    plt.xlabel("Iteración")
    plt.ylabel("Cantidad Promedio de Tiradas")
    plt.title(
        f"Cantidad Promedio de Jugadas en {agents} Agentes Montecarlo a Través de {games} Iteraciones"
    )

    plt.text(
        len(average_play_amounts) - len(average_play_amounts) * 0.05,
        np.min(average_play_amounts) * 0.98,
        f"Promedio de ultimas mil tiradas: {average_total_ult_mil:.2f}",
        fontsize=14,
        color="red",
        ha="right",
    )

    # plt.savefig(f"Random_{agents}_{games}.png")
    plt.show()


if __name__ == "__main__":
    main()

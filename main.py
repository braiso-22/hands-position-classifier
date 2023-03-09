from image_utils import Img, Camera
import mediapipe as mp
import pandas as pd
from pathlib import Path
from datetime import datetime


def mostrar_puntos_mano(frame, params):
    frame_num = params["frame_num"]

    hands = params["hands"]
    mp_hand = params["mp_hand"]
    mp_drawing = params["mp_drawing"]
    results = hands.process(image=frame)
    hand_landmarks = results.multi_hand_landmarks
    if hand_landmarks:
        for hand_landmark in hand_landmarks:
            mp_drawing.draw_landmarks(
                frame,
                hand_landmark,
                mp_hand.HAND_CONNECTIONS
            )

        if frame_num % 30 == 0:
            position_name = params["position_name"]
            df = pd.DataFrame(columns=["id", "x", "y"])
            for hand_landmark in hand_landmarks:
                for point_id, landmark in enumerate(hand_landmark.landmark):
                    # save to row of dataframe
                    df = df.append({
                        "id": point_id,
                        "x": landmark.x,
                        "y": landmark.y
                    }, ignore_index=True)
                    pass
            output_str = f"datasets/{datetime.now().strftime('%Y-%m-%d')}/{position_name}"
            # save to csv
            csv_output_dir = Path(output_str)
            csv_output_dir.mkdir(parents=True, exist_ok=True)
            csv_filename = f"{position_name}.csv"
            df.to_csv(csv_output_dir / csv_filename, mode='a', header=False, index=False)
            # save to image
            frames_output_dir = output_str + "/frames/"
            frames_output_dir_path = Path(output_str + "/frames/")
            frames_output_dir_path.mkdir(parents=True, exist_ok=True)
            frame_name = f"{position_name}_{frame_num}.jpg"
            Img.guardar_imagen(frame, (frames_output_dir + frame_name))

            pass
    Img.escribir(frame, pos=(20, 20), size=4, text=f"Frame: {frame_num}")
    return frame


def obtener_datos():
    position = input("Escribe un nombre para la posicion a entrenar:\n")
    mp_drawing = mp.solutions.drawing_utils
    mp_hand = mp.solutions.hands
    with mp_hand.Hands(
            max_num_hands=1,
            min_detection_confidence=0.3,
            min_tracking_confidence=0.6
    ) as hands:
        Camera.video_capture(
            operacion=mostrar_puntos_mano,
            operacion_params={
                "hands": hands,
                "mp_hand": mp_hand,
                "mp_drawing": mp_drawing,
                "position_name": position
            }
        )


def menu():
    print("1. Obtener datos de entrenamiento")
    print("2. Jugar")
    print("0. Salir")
    try:
        return int(input("Elija una opcion: "))
    except ValueError:
        return -1


def main():
    while True:
        opcion = menu()
        if opcion == 0:
            break
        elif opcion == 1:
            obtener_datos()
        elif opcion == 2:
            pass
        else:
            print("Opcion invalida")


if __name__ == '__main__':
    main()

from image_utils import Img, Camera
import mediapipe as mp
import pandas as pd
from pathlib import Path
from datetime import datetime
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

num = [i for i in range(0, 21)]
columns = [f"{j}{i}" for i in num for j in ["x", "y"]]
columns = ["frame", "position"] + columns


def guardar_dataframe_vacio():
    # check if file exists
    output_str = f"datasets/{datetime.now().strftime('%Y-%m-%d')}"
    csv_output_dir = Path(output_str)
    csv_output_dir.mkdir(parents=True, exist_ok=True)
    csv_filename = f"{datetime.now().strftime('%Y-%m-%d')}.csv"
    if (csv_output_dir / csv_filename).exists():
        return
    df = pd.DataFrame(columns=columns)
    df.to_csv(csv_output_dir / csv_filename, mode='a', header=True, index=False)


def crear_linea_df(df, hand_landmarks):
    for hand_landmark in hand_landmarks:
        for point_id, landmark in enumerate(hand_landmark.landmark):
            # where the frame is frame_num set x and y
            df.loc[f"x{point_id}"] = landmark.x
            df.loc[f"y{point_id}"] = landmark.y
    return df


def guardar_datos_frame(hand_landmarks, frame, position_name, frame_num):
    guardar_dataframe_vacio()
    df = pd.DataFrame(columns=columns)
    df = df.append({
        "frame": frame_num,
        "position": position_name,
    }, ignore_index=True)
    df = crear_linea_df(df, hand_landmarks)
    output_str = f"datasets/{datetime.now().strftime('%Y-%m-%d')}"
    # save to csv
    csv_output_dir = Path(output_str)
    csv_output_dir.mkdir(parents=True, exist_ok=True)
    csv_filename = f"{datetime.now().strftime('%Y-%m-%d')}.csv"
    df.to_csv(csv_output_dir / csv_filename, mode='a', header=False, index=False)
    # save to image
    frames_output_dir = output_str + "/frames/"
    frames_output_dir_path = Path(output_str + "/frames/")
    frames_output_dir_path.mkdir(parents=True, exist_ok=True)
    frame_name = f"{position_name}_{frame_num}.jpg"
    Img.guardar_imagen(frame, (frames_output_dir + frame_name))


def mostrar_puntos_mano_entrenar(frame, params):
    frame_num = params["frame_num"]

    hands = params["hands"]
    mp_hand = params["mp_hand"]
    mp_drawing = params["mp_drawing"]
    results = hands.process(image=frame)
    hand_landmarks = results.multi_hand_landmarks
    Img.escribir(frame, pos=(20, 20), size=4, text=f"Frame: {frame_num}")
    if not hand_landmarks:
        return frame
    for hand_landmark in hand_landmarks:
        mp_drawing.draw_landmarks(
            frame,
            hand_landmark,
            mp_hand.HAND_CONNECTIONS
        )

    if frame_num % 30 == 0:
        position_name = params["position_name"]
        guardar_datos_frame(
            hand_landmarks,
            frame,
            position_name,
            frame_num,
        )

    return frame


def mostrar_puntos_mano_jugar(frame, params):
    frame_num = params["frame_num"]

    hands = params["hands"]
    results = hands.process(image=frame)
    hand_landmarks = results.multi_hand_landmarks
    Img.escribir(frame, pos=(20, 20), size=4, text=f"Frame: {frame_num}")
    if not hand_landmarks:
        return frame
    modelo = params["modelo"]
    df = pd.DataFrame(columns=columns)
    df.loc["frame"] = 1
    df = crear_linea_df(df, hand_landmarks)
    df = df.drop(columns=["position"])
    df = df.drop(columns=["frame"])
    prediction = modelo.predict(df)
    Img.escribir(frame, pos=(20, 50), size=4, text=f"Prediction: {prediction[0]}")

    return frame


def entrenar_clasificador():
    df = pd.read_csv(f"datasets/{datetime.now().strftime('%Y-%m-%d')}/{datetime.now().strftime('%Y-%m-%d')}.csv")
    df = df.drop(columns=["frame"])
    y = df["position"]
    x = df.drop(columns=["position"])

    # Clasificador
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)
    clf = RandomForestClassifier(n_estimators=100)
    clf.fit(x_train, y_train)

    print(clf.score(x_test, y_test))
    return clf


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
            operacion=mostrar_puntos_mano_entrenar,
            operacion_params={
                "hands": hands,
                "mp_hand": mp_hand,
                "mp_drawing": mp_drawing,
                "position_name": position,
            }
        )


def jugar(modelo):
    mp_drawing = mp.solutions.drawing_utils
    mp_hand = mp.solutions.hands
    with mp_hand.Hands(
            max_num_hands=1,
            min_detection_confidence=0.3,
            min_tracking_confidence=0.6
    ) as hands:
        Camera.video_capture(
            operacion=mostrar_puntos_mano_jugar,
            operacion_params={
                "hands": hands,
                "mp_hand": mp_hand,
                "mp_drawing": mp_drawing,
                "modelo": modelo,
            }
        )


def menu():
    print("1. Obtener datos de entrenamiento")
    print("2. Entrenar clasificador")
    print("3. Jugar")
    print("0. Salir")
    try:
        return int(input("Elija una opcion: "))
    except ValueError:
        return -1


def main():
    modelo = None
    while True:
        opcion = menu()
        if opcion == 0:
            break
        elif opcion == 1:
            obtener_datos()
        elif opcion == 2:
            modelo = entrenar_clasificador()
            pass
        elif opcion == 3:
            if modelo is None:
                print("Primero debes entrenar el clasificador")
                continue
            jugar(modelo)
        else:
            print("Opcion invalida")


if __name__ == '__main__':
    main()

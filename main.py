import threading

from image_utils import Img, Camera
import mediapipe as mp
import joblib
import pandas as pd
from pathlib import Path
from datetime import datetime
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import recall_score
from sklearn.metrics import precision_score
from sklearn.tree import DecisionTreeClassifier
from selenium import webdriver
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.common.action_chains import ActionChains
import time

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


def crear_linea_df(df, frame_num, hand_landmarks):
    for hand_landmark in hand_landmarks:
        for point_id, landmark in enumerate(hand_landmark.landmark):
            # where the frame is frame_num set x and y
            df.loc[df.frame == frame_num, f"x{point_id}"] = landmark.x
            df.loc[df.frame == frame_num, f"y{point_id}"] = landmark.y
    return df


def guardar_datos_frame(hand_landmarks, frame, position_name, frame_num):
    guardar_dataframe_vacio()
    df = pd.DataFrame(columns=columns)
    df = df.append({
        "frame": frame_num,
        "position": position_name,
    }, ignore_index=True)
    df = crear_linea_df(df, frame_num, hand_landmarks)
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


def mostrar_prediccion(frame, params):
    frame_num = params["frame_num"]

    hands = params["hands"]
    results = hands.process(image=frame)
    hand_landmarks = results.multi_hand_landmarks
    Img.escribir(frame, pos=(20, 20), size=4, text=f"Frame: {frame_num}")
    if not hand_landmarks:
        return frame, None
    modelo: RandomForestClassifier = params["modelo"]
    df_new = pd.DataFrame(columns=columns)
    # new row
    df_new = df_new.append({
        "frame": frame_num,
        "position": "test",
    }, ignore_index=True)

    df_new = crear_linea_df(df_new, frame_num, hand_landmarks)
    df_new = df_new.drop(columns=["position"])
    df_new = df_new.drop(columns=["frame"])

    prediction = modelo.predict(df_new)

    Img.escribir(frame, pos=(20, 450), size=6, text=f"Prediction: {prediction[0]}")

    return frame, prediction


def execute_movement(key, action, duration):
    endtime = time.time() + (duration - 1) / 100
    while time.time() < endtime:
        action.key_down(key).perform()
    action.key_up(key).perform()


def camara_juego(frame, params):
    frame, prediccion = mostrar_prediccion(frame, params)
    # send key down to the game
    if prediccion is None:
        return frame, prediccion
    prediccion = prediccion[0]

    frame_num = params["frame_num"]
    frame_rate = 10
    if frame_num % frame_rate != 0:
        return frame, None
    controles = params["controles"]
    if prediccion == controles["izquierda"]:
        key = Keys.ARROW_LEFT
    elif prediccion == controles["derecha"]:
        key = Keys.ARROW_RIGHT
    elif prediccion == controles["arriba"]:
        key = Keys.SPACE
    else:
        return frame, None
    action: ActionChains = params["action_chains"]
    threading.Thread(target=execute_movement, args=(key, action, frame_rate)).start()

    return frame, None


def entrenar_clasificador():
    try:
        df = pd.read_csv(f"datasets/{datetime.now().strftime('%Y-%m-%d')}/{datetime.now().strftime('%Y-%m-%d')}.csv")
    except FileNotFoundError:
        print("No existen datos de entrenamiento")
        return None
    df = df.drop(columns=["frame"])
    y = df["position"]
    x = df.drop(columns=["position"])

    # Clasificador
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)
    clf = DecisionTreeClassifier(random_state=42)
    clf.fit(x_train, y_train)
    predictions = clf.predict(x_test)
    # see how many predictions were correct
    print("Accuracy:", accuracy_score(y_test, predictions))
    print("Precision:", precision_score(y_test, predictions, average="macro"))
    print("Recall:", recall_score(y_test, predictions, average="macro"))
    print("Confusion matrix:\n", confusion_matrix(y_test, predictions))
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


def prueba(modelo):
    if modelo is None:
        print("Primero debes entrenar el clasificador")
        return
    mp_drawing = mp.solutions.drawing_utils
    mp_hand = mp.solutions.hands
    with mp_hand.Hands(
            max_num_hands=1,
            min_detection_confidence=0.3,
            min_tracking_confidence=0.6
    ) as hands:
        Camera.video_capture(
            operacion=mostrar_prediccion,
            operacion_params={
                "hands": hands,
                "mp_hand": mp_hand,
                "mp_drawing": mp_drawing,
                "modelo": modelo,
            }
        )


def abrir_marcianitos() -> webdriver.Firefox:
    # With selenium open this webpage https://www.minijuegos.com/juego/space-invaders
    driver = webdriver.Firefox()
    driver.get('https://funhtml5games.com/spaceinvaders/index.html')

    return driver


def juego(modelo, controles):
    if modelo is None:
        print("Primero debes entrenar el clasificador")
        return
    if controles is None:
        print("Primero debes seleccionar los controles")
        return
    mp_drawing = mp.solutions.drawing_utils
    mp_hand = mp.solutions.hands
    with mp_hand.Hands(
            max_num_hands=1,
            min_detection_confidence=0.3,
            min_tracking_confidence=0.6
    ) as hands:
        driver = abrir_marcianitos()
        action_chains: ActionChains = ActionChains(driver)
        Camera.video_capture(
            operacion=camara_juego,
            operacion_params={
                "hands": hands,
                "mp_hand": mp_hand,
                "mp_drawing": mp_drawing,
                "modelo": modelo,
                "action_chains": action_chains,
                "controles": controles,
            }
        )


def seleccionar_controles(modelo):
    if modelo is None:
        print("Primero debes entrenar el clasificador")
        return None
    clases = modelo.classes_
    if len(clases) < 3:
        print("El modelo no tiene como minimo 3 poses a clasificar")
        return
    print("Tus poses son:")
    [print(f"{i + 1}. {clase}") for i, clase in enumerate(clases)]
    # ahora vamos a pedir que seleccione el orden de las clases
    controles = {"arriba": None, "izquierda": None, "derecha": None}
    for control in controles:
        while True:
            try:
                opcion = int(input(f"Selecciona la posicion de {control}: "))
                if opcion < 1 or opcion > len(clases):
                    print("Opcion no valida")
                    continue
                controles[control] = clases[opcion - 1]
                break
            except ValueError:
                print("Opcion no valida")
                continue
    return controles


def save_my_model(modelo):
    if modelo is None:
        print("Primero debes entrenar el clasificador")
        return
    nombre_modelo = input("Escribe un nombre para el modelo:\n")
    models_str = "./models"
    models_dir = Path(models_str)
    models_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(modelo, f"{models_str}/{nombre_modelo}.pkl")
    print(f"Modelo guardado como {nombre_modelo}.pkl")


def load_my_model():
    nombre_modelo = input("Escribe el nombre del modelo:\n")
    try:
        modelo = joblib.load(f"./models/{nombre_modelo}.pkl")
        print(f"Modelo {nombre_modelo}.pkl cargado correctamente")
        return modelo
    except FileNotFoundError:
        print("No existe un modelo con ese nombre")
        return None


def menu():
    print("1. Obtener datos de entrenamiento")
    print("2. Entrenar clasificador")
    print("3. Guardar clasificador")
    print("4. Cargar modelo guardado")
    print("5. Probar clasificador")
    print("6. Seleccionar controles")
    print("7. Jugar")
    print("0. Salir")
    try:
        return int(input("Elija una opcion: "))
    except ValueError:
        return -1


def execute_option(option, model, controls):
    if option == 0:
        exit(0)
    elif option == 1:
        obtener_datos()
    elif option == 2:
        model = entrenar_clasificador()
    elif option == 3:
        save_my_model(model)
    elif option == 4:
        model = load_my_model()
    elif option == 5:
        prueba(model)
    elif option == 6:
        seleccionar_controles(model)
    elif option == 7:
        juego(model, controls)
    else:
        print("Opcion invalida")
    return model, controls


def main():
    modelo = None
    controles = None

    while True:
        option = menu()
        modelo, controles = execute_option(option, modelo, controles)


if __name__ == '__main__':
    main()

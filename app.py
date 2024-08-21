from flask import Flask, request, jsonify
import tensorflow as tf
import numpy as np

app = Flask(__name__)

# Datos de entrenamiento
celcius = np.array([-40, -10, 0, 8, 15, 22, 38], dtype=float)
fahrenheit = np.array([-40, 14, 32, 46, 59, 72, 100], dtype=float)

# Definir modelo
oculta1 = tf.keras.layers.Dense(units=3, input_shape=[1])
oculta2 = tf.keras.layers.Dense(units=3)
salida = tf.keras.layers.Dense(units=1)
modelo = tf.keras.Sequential([oculta1, oculta2, salida])

modelo.compile(
    optimizer=tf.keras.optimizers.Adam(0.1),
    loss='mean_squared_error'
)

# Entrenar el modelo
print("--->Comenzando Entrenamiento...")
modelo.fit(celcius, fahrenheit, epochs=500, verbose=False)
print("Modelo Terminado!!!")

@app.route('/predict', methods=['POST'])
def predict():
    try:
        
        data = request.get_json(force=True)
        celcius_value = data.get('celcius')
        if celcius_value is None:
            return jsonify(error="El campo 'celcius' es obligatorio."), 400
        
        # Convertir el valor a un numpy array para la predicción
        celcius_array = np.array([[celcius_value]], dtype=float)
        fahrenheit_prediction = modelo.predict(celcius_array)
        
        # Convertir el valor de predicción a un tipo de dato nativo de Python
        fahrenheit_value = float(fahrenheit_prediction[0][0])
        
        return jsonify(fahrenheit=fahrenheit_value)
    except Exception as e:
        # Imprimir el error en la consola para depuración
        print(f"Error: {e}")
        return jsonify(error=str(e)), 500


if __name__ == '__main__':
    app.run(debug=True)

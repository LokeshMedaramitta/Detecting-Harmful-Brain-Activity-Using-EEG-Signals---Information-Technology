from flask import Flask, request, render_template, jsonify
import pickle
import numpy as np

app = Flask(__name__)

try:
    with open('model.pkl', 'rb') as model_file:
        model = pickle.load(model_file)
except Exception as e:
    print(f"Error loading the model: {e}")

@app.route('/')
def home():
    return render_template('dashboard.html')
@app.route('/test')
def test():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        if request.method == 'POST':
            int_features = [float(x) for x in request.form.values()]
            print("Received input features:", int_features)

            final_features = [np.array(int_features)]
            print("Final features for prediction:", final_features)
            
            prediction = model.predict_proba(final_features)
            print("Prediction:", prediction)

            output = '{0:.{1}f}'.format(prediction[0][1], 2)

            if prediction[0][1] > 0.5:
                result = 'HARMFUL BRAIN ACTIVITY DETECTED.'
            else:
                result = 'Your EEG data suggests a normal condition.'

            print("Result:", result, "Probability:", output)

            # Return a JSON response
            return jsonify({'prediction_text': f'{result}\nProbability: {output}'})

    except Exception as e:
        print("Error:", e)
        return jsonify({'error': 'An error occurred while processing the request.'})

if __name__ == '__main__':
    app.run(debug=True)

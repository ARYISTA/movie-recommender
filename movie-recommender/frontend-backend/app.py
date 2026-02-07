from flask import Flask, request, jsonify

app = Flask(__name__)

@app.route('/')
def home():
    return "Movie Recommender Backend Running"

@app.route('/recommend', methods=['POST'])
def recommend():
    data = request.json
    movie = data.get('movie')

    # Dummy recommendation for now
    recommendations = ["Inception", "Interstellar", "The Dark Knight"]

    return jsonify({
        "input_movie": movie,
        "recommended_movies": recommendations
    })

if __name__ == "__main__":
    app.run(debug=True)

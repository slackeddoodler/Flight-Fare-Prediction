import joblib
import gradio as gr
import numpy as np

pipeline = joblib.load("model.bin")

def predict_price(airline, source_city, departure_time, stops, arrival_time,
                  destination_city, duration, days_left, flight):
    
    sample = {
        "airline": airline,
        "source_city": source_city,
        "departure_time": departure_time,
        "stops": int(stops),
        "arrival_time": arrival_time,
        "destination_city": destination_city,
        "duration": float(duration),
        "days_left": int(days_left),
        "flight": flight
    }

    pred_log = pipeline.predict(sample)

    pred = np.expm1(pred_log)

    return round(float(pred), 2)


with gr.Blocks() as app:
    gr.Markdown("# Flight Price Predictor")

    with gr.Row():
        airline = gr.Dropdown(
            choices=["AirAsia", "Air_India", "GO_FIRST", "Indigo", "SpiceJet", "Vistara"],
            label="Airline"
        )
        source_city = gr.Dropdown(
            choices=["Delhi", "Mumbai", "Bangalore", "Kolkata", "Hyderabad", "Chennai"],
            label="Source City"
        )
        departure_time = gr.Dropdown(
            choices=["Early_Morning","Morning","Afternoon","Evening","Night","Late_Night"],
            label="Departure Time"
        )

    with gr.Row():
        stops = gr.Dropdown(
            choices=["0", "1", "2"], label="Number of stops"
        )
        arrival_time = gr.Dropdown(
            choices=["Early_Morning","Morning","Afternoon","Evening","Night","Late_Night"],
            label="Arrival Time"
        )
        destination_city = gr.Dropdown(
            choices=["Bangalore","Chennai","Delhi","Hyderabad","Kolkata","Mumbai"],
            label="Destination City"
        )

    with gr.Row():
        duration = gr.Number(label="Duration (hours)")
        days_left = gr.Number(label="Days left for departure")
        flight = gr.Textbox(label="Flight Code (e.g. SG-401, AI-983)")

    output = gr.Number(label="Predicted Price (INR)")

    submit = gr.Button("Predict Price")
    submit.click(
        predict_price,
        inputs=[airline, source_city, departure_time, stops, arrival_time, destination_city, duration, days_left, flight],
        outputs=output
    )

app.launch()

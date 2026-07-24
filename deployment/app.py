import gradio as gr
import spaces
import torch

from inference import solver
from examples import examples

def solve(prompt, a, b, c, d, e):
    options = [a, b, c, d, e]
    predictions = solver.predict(prompt, options)
    return predictions

demo = gr.Interface(
    fn=solve,
    inputs=[
        gr.Textbox(label="Question", placeholder="Enter the question here..."),
        gr.Textbox(label="Option A", placeholder="Enter option A here..."),
        gr.Textbox(label="Option B", placeholder="Enter option B here..."),
        gr.Textbox(label="Option C", placeholder="Enter option C here..."),
        gr.Textbox(label="Option D", placeholder="Enter option D here..."),
        gr.Textbox(label="Option E", placeholder="Enter option E here..."),
    ],
    outputs=[
        gr.Textbox(label="Top 3 Predictions"),
        gr.Textbox(label="Confidence"),
    ],
    title="Smart MCQ Solver",
    examples=examples,
    cache_examples=False,
    description="Smart MCQ solver can solve multiple choice questions. Enter the question and the options, and the model will predict the answer.",
    allow_flagging="never",
)

if __name__ == "__main__":
    demo.launch()

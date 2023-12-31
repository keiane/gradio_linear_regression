import gradio as gr
import numpy as np
import matplotlib.pyplot as plt

def regression(iterations, learning):
    # generate random data-set
    #np.random.seed(0) # choose random seed (optional)
    x = np.random.rand(100, 1)
    y = 2 + 3 * x + np.random.rand(100, 1)

    J = 0 # initialize J, this can be deleted once J is defined in the loop
    w = np.matrix([np.random.rand(),np.random.rand()]) # slope and y-intercept

    j_value = ""

    ## Write Linear Regression Code to Solve for w (slope and y-intercept) Here ##
    for p in range (iterations):
        for i in range(len(x)):
            # Calculate w and J here
            x_vec = np.matrix([x[i][0],1]) # Setting up a vector for x (x_vec[j] corresponds to w[j])
            # h = (define h here) ## Hint: you may need to transpose x or w by adding .T to the end of the variable
            h = w*x_vec.T
            # w = (define w update iteration here)
            w = w - (learning*(h - y[i])*x_vec)
            # J = (loss equation here)
            J = 0.5*(h - y[i])**2
            j_value += f"{str(i)}: {str(J)}\n"
        # print('Loss:', J)


    ## if done correctly the line should be in line with the data points ##

    #print(f"x: {x}\ny: {w[0,1] + (w[0,0] * x)}")
    equation = f"f = {w[0,0]}x + {w[0,1]}"

    fig = plt.figure()
    plt.scatter(x,y,s=10)
    plt.plot(x, w[0,1] + (w[0,0] * x), linestyle='solid')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.show()
    return equation, fig, gr.update(value=j_value)

with gr.Blocks() as demo:
    gr.Markdown("Linear Regression Code")
    # Row that takes in user input
    with gr.Row():
        ite = gr.Slider(label="Number of Iterations", minimum=1, maximum=100, step=1)
        learning = gr.Slider(label="Learning Rate Step Size", minimum=0.1, maximum=1, step=0.1)

    # Create a button to plot function and display equation
    with gr.Row():
        function = gr.Button("Plot Function")
        
    # Textbox that displays line equation    
    with gr.Row():
        y_box = gr.Textbox(label="Function f", interactive=False)

    # Print loss value in textbox
    with gr.Row():
        j_box = gr.Textbox(label="Loss, J", interactive=False)   
        plot = gr.Plot()
    
    demo.load(fn=regression, inputs=[ite, learning], outputs=[y_box, plot, j_box])
    function.click(fn=regression, inputs=[ite, learning], outputs=[y_box, plot, j_box])

if __name__ == "__main__":
    demo.launch()
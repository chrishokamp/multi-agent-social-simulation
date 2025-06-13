# Introduction

While researching the UI design for this project, I came across different potential implementations. Below is a textual summary and some resources linked to each option.

## Chatbot UI

Firstly, we can set up a simple chatbot UI which shows the message blobs of each agent interacting with one another. For example, if we have two agents like in the salesman problem, we can have the messages of the buyer appear on the left, and those of the salesman on the right.

**Resources:** Python: [Gradio](https://www.gradio.app) [Streamlit](https://streamlit.io), JavaScript: [ShadCN UI](https://ui.shadcn.com)

## 2D Game

Another option would be to create a 2D-based game UI, which looks similar to Pokemon or other classical games, where we display the actual agents as characters, and have the blobs appear above them, so essentially it would look like a 2D video game.

**Resources:** [Phaser](https://phaser.io) [PixiJS](https://pixijs.com)

## 3D Scene

Third, and definitely most complex option would be to implement a 3D scene, where we have the different agents represented as 3D objects, and show how they interact with each other etc. This option would be the coolest, but also the most time consuming, as we would have to first build our models using something like Blender.

**Resources:** [ThreeJS](https://threejs.org)

## Plotting Library Visualisation

Finally, to display the results and outcome of the multi-agent interaction after N simulations, we can use any visualisation/plotting library that JavaScript provides. Integrating this in Python would probably be easier using something like Matplotlib or Streamlit, but since the rest of our frontend will be built with JavaScript, it makes more sense to use a JS plotting library, to display any interesting statistics that we find, using heat maps, tree graphs etc.

**Resources:** Python: [Matplotlib](https://matplotlib.org) [Seaborn](https://seaborn.pydata.org) [Plotly](https://plotly.com), JavaScript: [D3 JS](https://d3js.org) [Recharts](https://recharts.org/en-US/)

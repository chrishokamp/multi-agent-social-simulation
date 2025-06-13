
From repository folder, cd research/Kitty then this command:
python3 -m http.server 8000
and then access
http://localhost:8000/index.html

You could also run the python script in autogen.ipynb to generate a new simulation and it will start the http server, although it will still fetch from the same conversation in negotiation_log_saved.json. 
The new simulation log overwrites the file negotiation_log.json. To see that in the webpage you can change the line in index.html to 
const response = await fetch('negotiation_log.json');
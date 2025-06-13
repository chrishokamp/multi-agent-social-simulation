[Return to index](../../index.md)

# Ollama - Command Line

## Chat with Ollama
Allows you to chat with Ollama in the termial.
```
ollama run <model>
```

## Start Ollama
Starts Ollama server. **Make sure that the desktop application is not running.**
```
ollama serve
```

## Create a Model
Models are created from Modelfiles.
```
ollama create mymodel -f ./Modelfile
```

## Remove a Model
```
ollama rm <model>
```

## Copy a Model
```
ollama cp <model> <name>
```

## List installed models
```
ollama list
```

## List loaded models
Lists models which are currently loaded.
```
ollama ps
```

## Stop a Model
Stop a model that is currently running.
```
ollama stop <model>
```

## Pull a Model
Can be used to download a new model or to update an existing 
model. In the case that the model is being updated, only the diff will be
pulled.
```
ollama pull <model>
```

## Show the modelfile of a specific model
```
ollama show --modelfile
```

[Return to index](../../index.md)
[Return to index](../../index.md)

# Modelfiles
 

## A simple example
Below is an example of a modelfile for creating a mario blueprint.
```
FROM llama3.2

# sets the temperature to 1 [higher is more creative, lower is more coherent]
PARAMETER temperature 1

# sets the context window size to 4096, this controls how many tokens the LLM can use as context to generate the next token
PARAMETER num_ctx 4096

# sets a custom system message to specify the behavior of the chat assistant
SYSTEM You are Mario from super mario bros, acting as an assistant.
```
Reference the [command line section](#command-line) for instructions on how to
create and run a model from a modelfile.

## Instructions

### FROM (required)
Defines the base model
```
FROM <model name>:<tag>
```
- [List of available base models](https://github.com/ollama/ollama#model-library)
- [Additional models](https://ollama.com/library)

### PARAMETER
Defines a parameter that can be set when the model is run.
```
PARAMETER <parameter> <value>
```
A full list of valid parameter values can be found [here](https://ollama.com/library)

### TEMPLATE
Full prompt template to be passed to the model. They use Go [template syntax](https://pkg.go.dev/text/template)

| Variable          | Description                                             |
| ----------------- | ------------------------------------------------------- |
| `{{ .System }}`   | The system message used to specify custom behaviour.    |
| `{{ .Prompt }}`   | The user prompt message.                                |
| `{{ .Response }}` | The response from the model. When generating a response, text after this variable is omitted. |

```
TEMPLATE """{{ if .System }}<|im_start|>system
{{ .System }}<|im_end|>
{{ end }}{{ if .Prompt }}<|im_start|>user
{{ .Prompt }}<|im_end|>
{{ end }}<|im_start|>assistant
"""
```

### SYSTEM
Specifies the system message to be used in the template, if applicable.
```
SYSTEM """<system message>"""
```

### ADAPTER
Specifies a fine tuned LoRA adapter that should apply to the base model. **If
the base model is not the same as the base model that the adapter was tuned from
then the behaviour will be erratic.
```
ADAPTER <path to safetensor adapter>
```

### LICENSE
Specifies legal license under which the model used with this Modelfile is shared
or distributed.
```
LICENSE """
<license text>
"""
```

### MESSAGE
Allows you to specift a message history for the model to use when responding.
Use multiple iterations of the MESSAGE command to build up a conversation.
```
MESSAGE <role> <message>
```
Valid roles:
| Role      | Description                                                   |
| --------- | ------------------------------------------------------------- |
| system    | Alternate way of providing the SYSTEM message for the model.  |
| user      | An example message of what the user could have asked.         |
| assistant | An example message of how the model should respond.           |

Example conversation:
```
MESSAGE user Is Toronto in Canada?
MESSAGE assistant yes
MESSAGE user Is Sacramento in Canada?
MESSAGE assistant no
MESSAGE user Is Ontario in Canada?
MESSAGE assistant yes
```

[Return to index](../../index.md)
import re

def parse_output_variables(message: str, output_variables: dict) -> tuple:
    results = {}
    
    lines = message.splitlines()
    filtered_lines = []
    
    for line in lines:
        if not any(re.search(rf"{vname}\s*=", line) for vname in output_variables):
            filtered_lines.append(line)
        
    full_match = True
    partial_match = False

    for vname, vtype in output_variables.items():
        str_pattern = rf"{vname}\s*=\s*\"(.*?)\""
        num_pattern = rf"{vname}\s*=\s*(\d+)"
        
        str_match = re.search(str_pattern, message)
        num_match = re.search(num_pattern, message)
        
        if vtype == "String":
            results[vname] = str_match.group(1) if str_match else None
        elif vtype == "Number":
            results[vname] = int(num_match.group(1)) if num_match else None

        if results[vname] == None:
            full_match = False
        else:
            partial_match = True
    
    return results, partial_match, full_match
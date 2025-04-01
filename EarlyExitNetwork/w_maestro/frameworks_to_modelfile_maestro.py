import re
import torch_maestro_summary as tms


def convert_to_model(model_params, model, outfile, device):

    if model_params['task'] in ['cifar100', 'cifar10']:
        INPUT_SIZE = tuple((int(d) for d in '3,32,32'.split(",")))
    elif model_params['task'] == 'tinyimagenet':
        INPUT_SIZE = tuple((int(d) for d in '3,64,64'.split(",")))
    else:
        raise ValueError(f"Unsupported task type: {model_params['task']}")

    model = model.to(device)
    mae_summary = tms.summary(model, INPUT_SIZE, device)
    linear_layer_numbers = []
    with open(outfile, "w") as fo:
        fo.write("Network {} {{\n".format(model.__module__))
        for key, val in mae_summary.items():

            # Extract the number after "Linear-"
            if key.startswith("Linear"):
                number = key.split("-")[-1]
                linear_layer_numbers.append(int(number))

            pc = re.compile("^Conv")
            pl = re.compile("^Linear")
            match_pc = pc.match(key)
            match_pl = pl.match(key)
            if match_pc or match_pl:
                fo.write("Layer {} {{\n".format(key))
                type = val["type"]
                fo.write("Type: {}\n".format(type))
                if not match_pl:
                    fo.write("Stride {{ X: {}, Y: {} }}\n".format(*val["stride"]))
                fo.write("Dimensions {{ K: {}, C: {}, R: {}, S: {}, Y: {}, X: {} }}\n".format(
                    *val["dimension_ic"][1:]))
                fo.write("}\n")
        fo.write("}")

    return linear_layer_numbers
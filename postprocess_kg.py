def remove_strings_from_file(file_path, target_strings):
    try:
        # Read the content of the file
        with open(file_path, "r") as file:
            content = file.read()

        # Remove each target string from the content
        for target_string in target_strings:
            content = content.replace(target_string, "")

        # Write the updated content back to the file
        with open(file_path, "w") as file:
            file.write(content)

        print(f"The specified strings have been removed from the file '{file_path}'.")

    except FileNotFoundError:
        print(f"The file '{file_path}' does not exist.")
    except Exception as e:
        print(f"An error occurred: {e}")


file_path_bgs = "data/bgs-hetero_733c98ba/bgs_repaired.owl"
file_path_aifb = "data/KGs/aifb.owl"
target_strings_aifb = ['<owl:Class rdf:about="http://www.w3.org/2002/07/owl#Thing"/>']
target_strings_bgs = [
    '<Class rdf:about="http://www.w3.org/1999/02/22-rdf-syntax-ns#ConceptScheme"/>',
    '<Class rdf:about="http://www.w3.org/1999/02/22-rdf-syntax-ns#Resource"/>',
]
remove_strings_from_file(file_path_bgs, target_strings_bgs)
remove_strings_from_file(file_path_aifb, target_strings_aifb)


# postprocessing of bgs file to add range to object property
from owlready2 import *

onto = get_ontology("data/bgs-hetero_733c98ba/bgs_repaired.owl").load()
with onto:
    sync_reasoner()

range_val = next(onto.classes())

for item in onto.object_properties():
    if len(item.range) < 1:
        item.range.append(range_val)

onto.save("data/KGs/bgs.owl")

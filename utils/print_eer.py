import os
import pickle
import pandas as pd

# Ensure pandas displays the entire DataFrame without truncation
pd.set_option("display.max_rows", None)  # No row limit
pd.set_option("display.max_columns", None)  # No column limit
pd.set_option("display.width", 1000)  # Adjust width to fit your terminal
pd.set_option("display.colheader_justify", "center")  # Center column headers for readability

base_dir = "scores"

# Morph types and models to populate the table
# morph_types = [ "amsl_MTCNN", "facemorpher_MTCNN", "opencv_MTCNN", "stylegan_MTCNN", "webmorph_MTCNN"]
# morph_types = ["greedy", "lma", "mipgan1", "mipgan2", "mordiff", "pipe"]
# dataset_name = "frill"
# printer = "digital"

morph_types = ["lmaubo"]
# morph_types = ["3D_morph"]
# dataset_name = "3DNTNU_Morphing_DB"
# dataset_name = "iPhone12_filled"
dataset_name = "digital"
printer = "digital"

# morph_types = ["lmaubo"]
# dataset_name = "synonot"

# morph_types = ["greedy", "lma", "lmaubo", "mipgan1", "mipgan2", "mordiff", "pipe"]
# dataset_name = "feret"

# Initialize a dictionary to store the data
eer_data = {}

# Iterate over the morph types to load corresponding .pkl files
for morph in morph_types:
    pkl_file = f"eer_table_{dataset_name}_{morph}_{printer}.pkl"
    # pkl_file = f"eer_table_{dataset_name}_{printer}.pkl"
    pkl_path = os.path.join(base_dir, pkl_file)

    if os.path.exists(pkl_path):
        with open(pkl_path, "rb") as file:
            data = pickle.load(file)
            for model_name, eer_value in data.items():
                if model_name not in eer_data:
                    eer_data[model_name] = {}
                eer_data[model_name][morph] = eer_value*100
                # eer_data[model_name]["avg_eer"] = eer_value*100
    else:
        print(f"Warning: {pkl_file} not found. Skipping.")

# Convert to a DataFrame for tabular representation
eer_df = pd.DataFrame(eer_data).T  # Transpose so models are rows
eer_df = eer_df.reindex(columns=morph_types)  # Ensure consistent column order
# eer_df = eer_df.reindex(columns=["avg_eer"])  # Ensure consistent column order

# Fill missing values with "N/A" or another placeholder
eer_df = eer_df.fillna("N/A")

# Print the table
print(f"\n EER FOR {dataset_name}")
print(eer_df)

# Optionally save as a CSV file
eer_df.to_csv(f"scores/eer_{dataset_name}_{printer}_summary_table.csv", index=True)


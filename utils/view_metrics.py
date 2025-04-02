import pandas as pd

# Read the CSV file
df = pd.read_csv('metrics_summary.csv')

# Define the protocol you're interested in (e.g., Protocol 3)
protocol = 'Protocol_3'
trainds = "iPhone11"
columns_to_select = ['Model'] + [col for col in df.columns if protocol in col]

# Filter the DataFrame to only include the relevant columns
filtered_df = df[columns_to_select]

# Reorganize columns by specifying the desired order
desired_order = ['Model',  # Keep 'Model' column first
                 [col for col in filtered_df.columns if col.endswith('EER')][0],  # Column ending in 'EER' second
                 [col for col in filtered_df.columns if col.endswith('BPCER@1')][0],  # Column ending in 'BPCER@1' third
                 [col for col in filtered_df.columns if col.endswith('BPCER@5')][0],  # Column ending in 'BPCER@5' fourth
                 [col for col in filtered_df.columns if col.endswith('BPCER@10')][0]]  # Column ending in 'BPCER@10' fifth

# Reorder the dataframe columns in filtered_df
filtered_df = filtered_df[desired_order]

# filtered_df = filtered_df[~filtered_df['Model'].str.startswith(f'spatial_channel_')]
# filtered_df = filtered_df[~filtered_df['Model'].str.startswith(f'spatial')]
# filtered_df = filtered_df[~filtered_df['Model'].str.startswith(f'channel')]


# fusion
filtered_df = filtered_df[
    (filtered_df['Model'].str.startswith(f'spatial_channel_{trainds}')) & 
    (filtered_df['Model'].str.endswith('_4_5')) |
    (filtered_df['Model'].str.startswith(f'spatial_channel_{trainds}')) & 
    (filtered_df['Model'].str.endswith('_4_5_mult')) |
    (filtered_df['Model'].str.startswith(f'spatial_channel_{trainds}')) & 
    (filtered_df['Model'].str.endswith('_4_5_add')) 
]
# # attn + img types
# filtered_df = filtered_df[
#     (filtered_df['Model'].str.startswith(f'spatial_channel_{trainds}')) & 
#     (filtered_df['Model'].str.endswith('_4_5')) |
#     (filtered_df['Model'].str.startswith(f'spatial_channel_{trainds}')) & 
#     (filtered_df['Model'].str.endswith('_color')) |
#     (filtered_df['Model'].str.startswith(f'spatial_channel_{trainds}')) & 
#     (filtered_df['Model'].str.endswith('_depth')) |
#     filtered_df['Model'].str.startswith(f'spatial_{trainds}') |
#     filtered_df['Model'].str.startswith(f'channel_{trainds}')
# ]

# filtered_df = filtered_df[filtered_df['Model'].str.startswith(f'spatial_channel_')]

# Print the filtered and reordered DataFrame
print(filtered_df)

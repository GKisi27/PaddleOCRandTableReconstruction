import pandas as pd

def remove_repeated_columns(df):
  from collections import defaultdict
  cols_to_remove = []
  # Count occurrences of identical columns (excluding index)
  for col1 in df.columns[:]:  # Start from second column (exclude index)
    if col1 in cols_to_remove:
      continue
    for col2 in df.columns[1:]:
      if col2 in cols_to_remove:
        continue
      if col1 != col2 and df[col1].equals(df[col2]):
        if col2 not in cols_to_remove:
          cols_to_remove.append(col2)
  df.drop(cols_to_remove, axis=1, inplace=True)
  return df

def check_column_similarity(df, col1_name, col2_name, threshold=0.6):
  col1 = df[col1_name]
  col2 = df[col2_name]
  if col1_name not in df.columns or col2_name not in df.columns:
    raise ValueError("Columns not found in DataFrame")
  elif len(col1) != len(col2):
    return False
  else:
    n_matches = (col1 == col2).sum()
    return n_matches / len(col1) >= threshold


def remove_similar_columns_func(df, thres):
  # Check similarity between specific columns
  similar_cols_to_remove = []
  for col1 in df.columns[:]:  # Start from second column (exclude index)
      if col1 in similar_cols_to_remove:
        continue
      for col2 in df.columns[1:]:
          if col2 in similar_cols_to_remove or col1 == col2:
            continue
          is_similar = check_column_similarity(df, col1, col2, thres)
          if is_similar:
            col1_total_data = df[col1].count()
            col2_total_data = df[col2].count()
            if col1_total_data>col2_total_data:
              similar_cols_to_remove.append(col2)
            else:
              similar_cols_to_remove.append(col1)
            
  df.drop(similar_cols_to_remove, axis=1, inplace=True)
  return df


# Apply the centering style
def highlight_centered(s):
    return ['text-align: center']*len(s)
  
# Function to replace consecutive duplicates with an empty string
def merge_consecutive_cells(df):
    df_merged = df.copy()
    for col in df_merged.columns:
        df_merged[col] = df_merged[col].mask(df_merged[col] == df_merged[col].shift(), '')
        
    styled_df = df_merged.style.apply(highlight_centered, axis=1)
    return styled_df

# if __name__ == "__main__":
#   df = pd.read_excel("result.xlsx", sheet_name = "Sheet1")

#   df_without_duplicates = remove_similar_columns(df.copy())
def rename_duplicate_columns(names):
    seen = {}
    renamed = []
    for name in names:
        if name not in seen:
            seen[name] = 0
            renamed.append(name)
        else:
            seen[name] += 1
            renamed.append(f"{name}_{seen[name]}")
    return renamed
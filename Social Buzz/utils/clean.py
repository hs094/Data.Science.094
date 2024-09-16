import pandas as pd

def clean(content_path: str, reactions_path: str):
    df_content = pd.read_csv(content_path)
    df_content.drop(labels='URL', axis=1, inplace=True)
    df_content.drop(labels='User ID', axis=1, inplace=True)
    # Replace " with empty string in 'Category' column
    df_content['Category'] = df_content['Category'].str.replace('"', '')
    
    # Rename the "Type" column to "Content Type"
    df_content.rename(columns={'Type': 'Content Type'}, inplace=True)

    df_reactions = pd.read_csv(reactions_path)
    df_reactions.rename(columns={'Type': 'Reaction Type'}, inplace=True)
    df_reactions.drop(labels='User ID', axis=1, inplace=True)
    # Drop rows in df_reactions where 'Content Type' is NA
    df_reactions.dropna(subset=['Reaction Type'], inplace=True)
    
    return df_content, df_reactions


df_content, df_reactions = clean('data/src/content.csv', 'data/src/reactions.csv')
print(df_content.head())
print(df_reactions.head())

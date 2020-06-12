def convert_clomuns_to_category(df, cat_list):
    for cat in cat_list:
        df[cat] = df[cat].astype('category')
    
    return df

def cat_to_one(encoder, df, col_list):
    for col in col_list:
        df[col] = encoder.fit(df[col])
        
    return df


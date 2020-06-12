def convert_to_type(df, cat_list, dtype):
    for cat in cat_list:
        df[cat] = df[cat].astype(dtype)
    
    return df

def cat_to_one(encoder, df, col_list):
    for col in col_list:
        df[col] = encoder.fit(df[col])
        
    return df

def na_to_zero(df, col_list):
    for col in col_list:
        df[col] = df[col].fillna(0)

    return df




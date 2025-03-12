import streamlit as st
import pandas as pd
import os
import re
import requests
from bs4 import BeautifulSoup
import json

# 1) Optional: For JavaScript rendering, you might do:
# from requests_html import HTMLSession

###########################
# AI Parsing (OpenAI GPT)
###########################
# Make sure you have `openai` in your requirements.txt (e.g. "openai>=0.27.0")
# Then set OPENAI_API_KEY as an environment variable, or store your key securely.
import openai
openai.api_key = os.environ.get("OPENAI_API_KEY", "")

def ai_extract_fields(raw_text):
    """
    Sends the raw text to OpenAI GPT to extract structured fields.
    Modify the system/user prompts to get the fields you want.
    Returns a dict of {field_name: value}.
    """
    if not openai.api_key:
        # If no key, just return empty
        return {"error": "OpenAI API key not set. Cannot parse via AI."}

    # Example system & user prompts. Adjust to your liking.
    # We'll attempt to parse address, city, price, description, etc.
    system_prompt = (
        "You are an AI that extracts structured data from real estate listing text. "
        "Output your final response in valid JSON only. No extra commentary."
    )

    user_prompt = f"""
Given the following property text, extract the following fields if present:
- address
- city
- region
- postal_code
- listing_type (e.g. 'Lot', 'Duplex', 'Condo', 'Single-family', etc.)
- listing_price
- number_of_bedrooms
- number_of_bathrooms
- property_size (lot size)
- year_built
- real_estate_broker
- phone_number
- any other relevant info

If a field is unknown or not mentioned, leave it empty or null.

Text to parse:
{raw_text}

Format your response strictly as valid JSON object. 
Example:
{{
    "address": "...",
    "city": "...",
    "region": "...",
    "postal_code": "...",
    "listing_type": "...",
    "listing_price": "...",
    "number_of_bedrooms": "...",
    "number_of_bathrooms": "...",
    "property_size": "...",
    "year_built": "...",
    "real_estate_broker": "...",
    "phone_number": "...",
    "other_notes": "..."
}}
"""
    try:
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",  # or "gpt-4" if you have access
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            temperature=0.1
        )
        content = response["choices"][0]["message"]["content"]
        # Attempt to parse JSON
        data = json.loads(content)
        return data

    except Exception as e:
        return {"error": f"AI parsing failed: {e}"}


#############################
# Fetching Page and Extracting Text
#############################
def fetch_page_text(url):
    headers = {
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/110.0.0.0 "
            "Safari/537.36"
        )
    }

    try:
        response = requests.get(url, headers=headers, timeout=10)
        response.raise_for_status()
    except Exception as e:
        return None, f"Error fetching the URL: {e}"

    soup = BeautifulSoup(response.text, "html.parser")
    all_text = soup.get_text(separator=" ", strip=True)
    return all_text, None

# If needed, for JavaScript rendering, you might do:
# def fetch_page_text(url):
#     session = HTMLSession()
#     headers = {"User-Agent": "Mozilla/5.0 ..."}
#     r = session.get(url, headers=headers)
#     r.html.render(timeout=20)
#     all_text = r.html.text
#     return all_text, None

#############################
# CSV Data
#############################
def load_data(csv_path="data.csv"):
    if os.path.exists(csv_path):
        return pd.read_csv(csv_path, keep_default_na=False)
    else:
        # Create an empty DataFrame with some default columns
        columns = [
            "URL", 
            "FullText", 
            "address",
            "city",
            "region",
            "postal_code",
            "listing_type",
            "listing_price",
            "number_of_bedrooms",
            "number_of_bathrooms",
            "property_size",
            "year_built",
            "real_estate_broker",
            "phone_number",
            "other_notes",
            "Notes",  # your custom note
            "Status"  # for 'Interested' or not
        ]
        df = pd.DataFrame(columns=columns)
        return df

def save_data(df, csv_path="data.csv"):
    df.to_csv(csv_path, index=False)

#############################
# Main Streamlit App
#############################
def main():
    st.title("AI-Powered Webpage Scraper & Property Parser")

    df = load_data()

    # Ensure "Notes" and "Status" exist
    if "Notes" not in df.columns:
        df["Notes"] = ""
    if "Status" not in df.columns:
        df["Status"] = ""

    st.header("Add a new property by URL or text containing a Centris link")

    user_input = st.text_area(
        "Paste a Centris link (or any text with a link):",
        placeholder="Example: https://www.centris.ca/en/lots~for-sale~baie-d-urfe/20791857"
    )

    ai_enabled = openai.api_key != ""
    if not ai_enabled:
        st.warning("No OpenAI API key detected. We'll still fetch text, but AI parsing won't work.")

    if st.button("Fetch & Parse via AI"):
        # Extract the URL from the input
        url_pattern = r'(https?://[^\s]+)'
        match = re.search(url_pattern, user_input)
        if not match:
            st.error("No URL found in the given input.")
        else:
            url = match.group(0)
            st.write(f"Fetching text from: {url}")

            all_text, error_msg = fetch_page_text(url)
            if error_msg:
                st.error(error_msg)
            else:
                if not all_text:
                    st.warning("Fetched the page, but no text found.")
                    return

                st.write("Successfully got page text. Using AI to parse relevant fields...")
                if ai_enabled:
                    field_data = ai_extract_fields(all_text)
                else:
                    field_data = {"error": "AI not available; no API key configured."}

                if "error" in field_data:
                    st.error(f"AI Error: {field_data['error']}")
                else:
                    # Convert field_data to a single row DataFrame
                    new_row = {"URL": url, "FullText": all_text}
                    for k, v in field_data.items():
                        new_row[k] = v

                    # Ensure any new columns exist in the main DF
                    for col in new_row.keys():
                        if col not in df.columns:
                            df[col] = None

                    df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)
                    save_data(df)
                    st.success("Page text + AI-parsed data appended to CSV!")

    # Filtering
    st.header("Filter the Data")
    filter_column = st.selectbox("Select a column to filter on", ["(No filter)"] + list(df.columns), index=0)
    filtered_df = df.copy()

    if filter_column != "(No filter)":
        if pd.api.types.is_numeric_dtype(filtered_df[filter_column]):
            min_val = float(filtered_df[filter_column].min()) if not filtered_df[filter_column].empty else 0
            max_val = float(filtered_df[filter_column].max()) if not filtered_df[filter_column].empty else 0
            user_range = st.slider("Select range", min_val, max_val, (min_val, max_val))
            filtered_df = filtered_df[
                (filtered_df[filter_column] >= user_range[0]) &
                (filtered_df[filter_column] <= user_range[1])
            ]
        else:
            txt_filter = st.text_input("Enter text to search in this column")
            if txt_filter:
                filtered_df = filtered_df[filtered_df[filter_column].str.contains(txt_filter, case=False, na=False)]

    st.subheader("Editable Table (Notes / Status, etc.)")
    editable_cols = [col for col in filtered_df.columns]
    edited_df = st.data_editor(filtered_df[editable_cols], use_container_width=True, num_rows="dynamic")

    if st.button("Save Edits to CSV"):
        for i in edited_df.index:
            url_val = edited_df.at[i, "URL"] if "URL" in edited_df.columns else None
            if url_val in df["URL"].values:
                main_idx = df.index[df["URL"] == url_val][0]
                for col in editable_cols:
                    df.at[main_idx, col] = edited_df.at[i, col]
            else:
                new_row_series = edited_df.loc[i]
                df = pd.concat([df, pd.DataFrame([new_row_series])], ignore_index=True)

        save_data(df)
        st.success("Edits saved to CSV.")

    st.subheader("Export Table")
    csv_all = df.to_csv(index=False).encode('utf-8')
    st.download_button(
        label="Download Full CSV",
        data=csv_all,
        file_name="all_properties.csv",
        mime="text/csv"
    )

    csv_filtered = filtered_df.to_csv(index=False).encode('utf-8')
    st.download_button(
        label="Download Filtered CSV",
        data=csv_filtered,
        file_name="filtered_properties.csv",
        mime="text/csv"
    )

if __name__ == "__main__":
    main()

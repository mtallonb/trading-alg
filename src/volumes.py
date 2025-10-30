import glob
import os

import krakenex
import pandas as pd

from utils.basic import get_new_prices

# --- Configuración de Directorios ---
PRICES_DIR = './data/prices'
OHLCV_DIR = './data/OHLCV_prices/'
OUTPUT_DIR = './data/prices_with_volume/'
HEADER_PRICES = ["TIMESTAMP", "O", "H", "L", "C", "VOL", "TRADES"]
# ------------------------------------
kapi = krakenex.API()
kapi.load_key('./data/keys/kraken.key')


print(f"Searching for files in: {PRICES_DIR}")

excluded_assets = ['MATICEUR']

# 2. Find all daily price files
# glob.glob finds files matching the pattern
price_files = glob.glob(os.path.join(PRICES_DIR, '*_CLOSE_DAILY.csv'))

if not price_files:
    print(f"Error: No '*_CLOSE_DAILY.csv' files found in the directory '{PRICES_DIR}'.")
    print("Make sure the script is in the correct location.")
    exit(1)

print(f"Found {len(price_files)} price files. Processing...")

# 3. Iterate over each price file found
for price_file_path in price_files:
    try:
        # 4. Extract the asset name from the filename
        # e.g., 'prices/AAVEEUR_CLOSE_DAILY.csv' -> 'AAVEEUR'
        base_name = os.path.basename(price_file_path)
        asset_name = base_name.replace('_CLOSE_DAILY.csv', '')
        if asset_name in excluded_assets:
            continue

        print(f"\n--- Processing Asset: {asset_name} ---")

        # 5. Build the path to the corresponding OHLC file
        ohlc_file_path = os.path.join(OHLCV_DIR, f'{asset_name}_1440.csv')

        # 6. Load the two CSV files into pandas DataFrames

        # 6a. Load the price file (Source 1)
        # Format: DATE,PRICE
        df_price = pd.read_csv(price_file_path)
        # print(f"  > Read {price_file_path} ({len(df_price)} rows)")

        # 6b. Load the OHLC file (Source 2)
        # 'names=HEADER_PRICES' is used because the file has no header
        df_ohlc = pd.read_csv(ohlc_file_path, names=HEADER_PRICES)
        # print(f"  > Read {ohlc_file_path} ({len(df_ohlc)} rows)")

        # 7. Data Transformation (The key step)

        # 7a. Convert the TIMESTAMP (seconds) to a datetime object
        # 'unit='s'' tells pandas the number is a Unix timestamp in seconds
        datetime_obj = pd.to_datetime(df_ohlc['TIMESTAMP'], unit='s')

        # 7b. Convert that datetime to a 'YYYY-MM-DD' string
        # .dt.strftime('%Y-%m-%d') formats the date as text
        # This creates the new 'DATE' column we will use for the join
        df_ohlc['DATE'] = datetime_obj.dt.strftime('%Y-%m-%d')

        # print("  > Timestamp converted to YYYY-MM-DD format.")

        # 8. Merge the two DataFrames

        # We select only the columns we need from OHLC: 'DATE' and 'VOL'
        df_volume = df_ohlc[['DATE', 'VOL']]

        # pd.merge joins df_price with df_volume using the 'DATE' column as the key
        # how='left': Keeps all rows from df_price (the left side) and adds 'VOL'
        #            if it finds a matching date. If not, it puts 'NaN' (Null).
        df_merged = pd.merge(df_price, df_volume, on='DATE', how='left')

        # Fill gaps on volume as NaN
        # --- INICIO: RELLENAR HUECOS DE VOLUMEN USANDO LA API DE KRAKEN ---
        if df_merged['VOL'].isnull().any():
            print("  > Se han encontrado datos de volumen faltantes. Rellenando desde la API de Kraken...")

            # 1. Encontrar la primera fecha donde 'VOL' es NaN
            first_nan_date_str = df_merged[df_merged['VOL'].isnull()]['DATE'].iloc[0]
            since_timestamp = int(pd.to_datetime(first_nan_date_str).timestamp())

            print(f"  > Obteniendo datos para {asset_name} desde {first_nan_date_str}")

            try:
                # 2. Llamar a la API de Kraken
                ohlcv = get_new_prices(
                    kapi=kapi,
                    asset_name=asset_name,
                    timestamp_from=since_timestamp,
                    with_volumes=True,
                )

                if not ohlcv.empty:
                    datetime_obj = pd.to_datetime(ohlcv['TIMESTAMP'], unit='s')
                    ohlcv['DATE'] = datetime_obj.dt.strftime('%Y-%m-%d')
                    ohlcv['VOL'] = pd.to_numeric(ohlcv['VOL'])

                    # 4. Rellenar los valores NaN en df_merged
                    df_new_volume = ohlcv[['DATE', 'VOL']]

                    # Preparamos los dataframes para la actualización
                    df_merged.set_index('DATE', inplace=True)
                    df_new_volume.set_index('DATE', inplace=True)

                    # Actualizamos los valores faltantes en df_merged
                    df_merged.update(df_new_volume)

                    # Restauramos el índice
                    df_merged.reset_index(inplace=True)

                    print(f"  > Éxito: Se rellenaron {len(df_new_volume)} valores de volumen.")
                else:
                    print(
                        f"  > La API de Kraken no devolvió datos nuevos para {asset_name} desde {first_nan_date_str}.",
                    )

            except Exception as api_error:
                print(f"  > ERROR al contactar con la API de Kraken para {asset_name}: {api_error}")
        # --- FIN DEL BLOQUE PARA RELLENAR HUECOS ---

        # 9. Save the result
        output_filename = os.path.join(OUTPUT_DIR, f'{asset_name}_DAILY_WITH_VOLUME.csv')

        # index=False: Don't save the pandas index in the CSV
        df_merged['VOL'] = df_merged['VOL'].round(2)
        df_merged.to_csv(output_filename, index=False)

        print(f"  > Success! Merged file saved to: {output_filename}")

    except FileNotFoundError:
        # Handle the case where the price file exists, but the OHLC file does not
        print(f"  > ERROR: Corresponding OHLC file not found at: {ohlc_file_path}")
        print(f"  > Skipping asset {asset_name}.")
    except Exception as e:
        # Catch any other unexpected error
        print(f"  > ERROR: An unexpected error occurred while processing {asset_name}: {e}")

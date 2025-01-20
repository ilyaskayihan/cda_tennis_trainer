import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.spatial import ConvexHull
import matplotlib

matplotlib.use('Agg')  # Kompatibilität mit Streamlit

# Verfügbare Dateipfade
file_paths = {
    "Schlieren_20231019_Labels.xlsx": r"C:\Users\ilyas\OneDrive\Desktop\cda2_versuch3000\Daten\Schlieren_20231019_Labels.xlsx",
    "Schlieren_20231019_Labels2.xlsx": r"C:\Users\ilyas\OneDrive\Desktop\cda2_versuch3000\Daten\Schlieren_20231019_Labels2.xlsx",
    "Schlieren_20231019_Labels3.xlsx": r"C:\Users\ilyas\OneDrive\Desktop\cda2_versuch3000\Daten\Schlieren_20231019_Labels3.xlsx"
}

# Kalibrierungspunkte
P1 = np.array([1.763, 0.110, 16.476])
P2 = np.array([9.341, 0.435, 16.512])
P3 = np.array([-1.079, 0.430, 8.327])

# Basisvektoren und Transformation
r1 = P2 - P1
r2 = -(P3 - 0.5 * (P1 + P2))
r1u = r1 / np.linalg.norm(r1)
r2u = r2 / np.linalg.norm(r2)
r3u = np.cross(r1u, r2u)
T0 = 0.5 * (P1 + P2) + 0.25 * r2u
R = np.array([r1u, r2u, r3u]).T


# Transformation Funktion
def transform_coordinates(x, y, z):
    original_coords = np.array([x, y, z]) - T0
    transformed_coords = R.T @ original_coords
    return transformed_coords


# Daten einlesen und vorbereiten
@st.cache_data
def load_data(selected_file):
    file_path = file_paths[selected_file]
    df = pd.read_excel(file_path)
    df[['x', 'y', 'z']] = df.apply(lambda row: transform_coordinates(row['x'], row['y'], row['z']), axis=1,
                                   result_type='expand')
    df_person = df[df['Objekt'] == 0].copy()

    df_person[['vx', 'xy', 'vz']] = df_person[['vx', 'xy', 'vz']].apply(pd.to_numeric, errors='coerce')
    df_person['speed'] = np.sqrt(df_person['vx'] ** 2 + df_person['xy'] ** 2 + df_person['vz'] ** 2)

    Q1, Q3 = df_person['speed'].quantile(0.25), df_person['speed'].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound, upper_bound = Q1 - 1.5 * IQR, Q3 + 1.5 * IQR
    df_cleaned = df_person[(df_person['speed'] >= lower_bound) & (df_person['speed'] <= upper_bound)]

    df_cleaned[['x', 'y']] = df_cleaned[['x', 'y']].apply(pd.to_numeric, errors='coerce')
    df_cleaned['direction'] = np.arctan2(df_cleaned['y'].diff(), df_cleaned['x'].diff())
    df_cleaned['angular_change'] = np.abs(np.diff(df_cleaned['direction'], prepend=df_cleaned['direction'].iloc[0]))

    return df_cleaned


# Funktion zur Darstellung des Tennisfelds
def draw_tennis_court(x_min, x_max, y_min, y_max):
    plt.plot([x_min, x_min, x_max, x_max, x_min],
             [y_min, y_max, y_max, y_min, y_min], 'k-', linewidth=2)
    plt.plot([(x_min + x_max) / 2, (x_min + x_max) / 2],
             [y_min, y_max], 'k--', linewidth=1)
    plt.plot([x_min, x_max],
             [(y_min + y_max) / 2, (y_min + y_max) / 2], 'k--', linewidth=1)
    plt.xlabel("x-Position (Meter)")
    plt.ylabel("y-Position (Meter)")
    plt.grid()


# Start des Dashboards
def main():
    st.title("Tennis-Tracking-Dashboard")

    # Auswahl der Datei in der Seitenleiste
    st.sidebar.header("Daten auswählen")
    selected_file = st.sidebar.selectbox("Wähle einen Datensatz", list(file_paths.keys()))

    # Daten laden basierend auf der Auswahl
    df_cleaned = load_data(selected_file)

    # Navigation für Seiten
    st.sidebar.header("Navigation")
    page = st.sidebar.selectbox("Wähle eine Seite", ["Datenübersicht", "Visualisierungen"])

    if page == "Datenübersicht":
        st.header("Datenübersicht")
        st.write(f"**Angezeigte Datei:** {selected_file}")
        st.dataframe(df_cleaned.head(10))
        st.write("Statistische Übersicht:")
        st.write(df_cleaned[['speed', 'vx', 'xy', 'vz']].describe())

    elif page == "Visualisierungen":
        st.header("Visualisierungen")

        plot_option = st.sidebar.selectbox("Wähle eine Plot-Art",
                                           ["Geschwindigkeit", "Position", "Bewegungsrichtung"])

        if plot_option == "Geschwindigkeit":
            st.subheader("Geschwindigkeit über die Zeit ohne Ausreißer")
            fig, ax = plt.subplots(figsize=(10, 5))
            ax.plot(df_cleaned['Frame'], df_cleaned['speed'], label='Geschwindigkeit', color='blue')
            ax.set_xlabel('Frame')
            ax.set_ylabel('Geschwindigkeit (m/s)')
            ax.legend()
            ax.grid()
            st.pyplot(fig)

            st.subheader("Boxplot der Geschwindigkeit ohne Ausreißer")
            fig, ax = plt.subplots(figsize=(6, 5))
            ax.boxplot(df_cleaned['speed'].dropna(), vert=False, patch_artist=True,
                       boxprops=dict(facecolor='green', color='black'))
            ax.set_xlabel('Geschwindigkeit (m/s)')
            ax.grid()
            st.pyplot(fig)

        elif plot_option == "Position":
            st.subheader("Position der Person auf dem Tennisfeld")

            fig, ax = plt.subplots(figsize=(10, 6))
            draw_tennis_court(df_cleaned['x'].min(), df_cleaned['x'].max(), df_cleaned['y'].min(), df_cleaned['y'].max())
            sc = ax.scatter(df_cleaned['x'], df_cleaned['y'], c=df_cleaned['speed'], cmap='coolwarm', edgecolors='k')
            plt.colorbar(sc, label='Geschwindigkeit (m/s)')
            plt.grid()
            st.pyplot(fig)

            st.subheader("Heatmap der Positionen")
            fig, ax = plt.subplots(figsize=(10, 6))
            draw_tennis_court(df_cleaned['x'].min(), df_cleaned['x'].max(), df_cleaned['y'].min(), df_cleaned['y'].max())
            sns.kdeplot(x=df_cleaned['x'], y=df_cleaned['y'], cmap="YlOrBr", fill=True, levels=50, thresh=0, ax=ax)
            st.pyplot(fig)

        elif plot_option == "Bewegungsrichtung":
            st.subheader("Bewegungsrichtung (Polar-Histogramm)")
            fig, ax = plt.subplots(figsize=(6, 6), subplot_kw={'projection': 'polar'})
            ax.hist(df_cleaned['direction'].dropna(), bins=36, color='purple', alpha=0.7)
            ax.set_title('Häufigkeit der Bewegungsrichtungen')
            st.pyplot(fig)

            st.subheader("Richtungsänderung über die Zeit")
            fig, ax = plt.subplots(figsize=(10, 5))
            ax.plot(df_cleaned['Frame'], df_cleaned['angular_change'], label='Richtungsänderung', color='purple')
            ax.set_xlabel('Frame')
            ax.set_ylabel('Winkeländerung (radian)')
            ax.legend()
            ax.grid()
            st.pyplot(fig)


if __name__ == "__main__":
    main()

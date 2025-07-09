import streamlit as st # Importing Streamlit for web app development
import json # Importing JSON for handling JSON data
from prophet.serialize import model_from_json # Importing Prophet for time series forecasting
import pandas as pd # Importing Pandas for data manipulation
from prophet.plot import plot_plotly # Importing Plotly for interactive plotting

# Function to load the Prophet model from a JSON file
def load_model():
    with open('modelo_O3_prophet.json', 'r') as file_in:
        modelo = model_from_json(json.load(file_in))
        return modelo

modelo = load_model()  # Load the Prophet model from JSON file

# Layout of the Streamlit app
st.title('Previsão de Níveis de Ozônio (O3) Utilizando a Biblioteca Prophet') # Title of the app

# Adding a text input to describe the project
st.caption('''Este projeto utiliza a biblioteca Prophet para prever os níveis de ozônio (O3) em ug/m3. O modelo
           criado foi treinado com dados até o dia 05/05/2023 e possui um erro de previsão (RMSE - Erro Quadrático Médio) igual a 17.43 nos dados de teste.
           O usuário pode inserir o número de dias para os quais deseja a previsão, e o modelo gerará o gráfico
           iterativo contendo as estimativas baseadas em dados históricos de concentração de O3.
           Além disso, uma tabela será exibida com os valores estimados para cada dia.''')

st.subheader('Insira o número de dias para previsão:') # Subheader for input section

dias = st.number_input('', min_value=1, value=1, step=1)  # Input for number of days to forecast

if 'previsao_feita' not in st.session_state: # Initialize session state variable
    st.session_state['previsao_feita'] = False # Variable to track if forecast has been made
    st.session_state['dados_previsao'] = None  # Variable to store forecast data

if st.button('Prever'):  # Button to trigger the forecast
    st.session_state.previsao_feita = True # Update session state to indicate forecast has been made
    futuro = modelo.make_future_dataframe(periods=dias, freq='D')  # Create future dataframe for prediction
    previsao = modelo.predict(futuro)  # Generate forecast using the model
    st.session_state['dados_previsao'] = previsao  # Store forecast data in session state

if st.session_state.previsao_feita:  # Check if forecast has been made
    fig = plot_plotly(modelo, st.session_state['dados_previsao']) # Create interactive plot
    # Display the forecast data in a table with white background and black text
    fig.update_layout({
        'plot_bgcolor': 'rgba(255, 255, 255, 1)', # Set background color of the plot
        'paper_bgcolor': 'rgba(255, 255, 255, 1)', # Set paper background color
        'title': {'text': "Previsão de Ozônio", 'font': {'color': 'black'}}, # Set title and font color
        'xaxis': {'title': 'Data', 'title_font': {'color': 'black'}, 'tickfont': {'color': 'black'}}, # Set x-axis title and font color
        'yaxis': {'title': 'Nível de Ozônio (O3 ug/m3)', 'title_font': {'color': 'black'}, 'tickfont': {'color': 'black'}} # Set y-axis title and font color
    })
    st.plotly_chart(fig) # Display the plot in the app

    previsao = st.session_state['dados_previsao'] # Retrieve forecast data from session state
    tabela_previsao = previsao[['ds', 'yhat']].tail(dias) # Select relevant columns for the table
    tabela_previsao.columns = ['Data (Dia/Mês/Ano)', 'Nível de Ozônio (O3 ug/m3)'] # Rename columns for clarity
    tabela_previsao['Data (Dia/Mês/Ano)'] = tabela_previsao['Data (Dia/Mês/Ano)'].dt.strftime('%d/%m/%Y') # Format date for display
    tabela_previsao['Nível de Ozônio (O3 ug/m3)'] = tabela_previsao['Nível de Ozônio (O3 ug/m3)'].round(2) # Round values for better readability
    tabela_previsao.reset_index(drop=True, inplace=True) # Reset index for the table
    st.write('Tabela contendo as previsões de ozônio (ug/m3) para os próximos {} dias:'.format(dias)) # Display table header
    st.dataframe(tabela_previsao, height=300) # Display the forecast table with specified height

    csv = tabela_previsao.to_csv(index=False)  # Convert the table to CSV format
    st.download_button(label='Baixar tabela como csv', data=csv, file_name='previsao_ozonio.csv', mime='text/csv') # Button to download the table as CSV


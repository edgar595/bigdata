from flask import Flask, render_template, request, redirect
from dash import Dash, html, dcc
from dash.dependencies import Input, Output, State
import plotly.graph_objs as go
import plotly.express as px
from dash.exceptions import PreventUpdate
import pandas as pd
import dash_table
import math
import datetime
import joblib

# Load the transformed DataFrame (replace with your data source)
file_path = "C:/Users/USER/OneDrive/Desktop/data analysis/Python/bigdata/transformed_data.xlsx"
transformed_df = pd.read_excel(file_path)

# Columns to be used for prediction
selected_columns = ['Average_Goals_Scored', 'Average_Goals_Conceded', 'Elo Rating']

# Load the trained model
model = joblib.load('model')

# Define the mapping for points
point_mapping = {'W': 3, 'D': 1, 'L': -1}

# Apply the mapping to create the "Plot Points" column
transformed_df['Plot Points'] = transformed_df['Result'].map(point_mapping)

# Define the start date
start_date = datetime.datetime(2023, 8, 8)

# Define a list of unique teams in the DataFrame
unique_teams = transformed_df['Team'].unique()

# Create a dropdown for team selection
team_dropdown = dcc.Dropdown(
    id='team-dropdown',
    options=[{'label': team, 'value': team} for team in unique_teams],
    value='Chelsea'  # Set the default selected team
)


# Create a Flask instance
app_flask = Flask(__name__)

# Create a list of unique countries in your data
unique_countries = transformed_df['Country'].unique()

# Define a route and view function for the login page
@app_flask.route('/')
def login():
    return render_template('security.html')

# Define a route and view function to handle form submission
@app_flask.route('/authenticate', methods=['POST'])
def authenticate():
    security_key = request.form.get('security_key')
    # Check if the security_key is correct (e.g., compare it to '1234')
    if security_key == '1234':
        # Redirect to the Dash app
        return redirect('/dashboard')
    else:
        # Add an error message or redirect to an error page if the key is incorrect
        return "Access Denied: Invalid Security Key"

# Create a Dash app instance
dash_app = Dash(__name__, server=app_flask, url_base_pathname='/dashboard/')

# Define the Dash app layout
dash_app.layout = html.Div([
    html.H2("Football Prediction", style={'text-align': 'center'}),

    html.H2("Introduction"),
    html.H4("""
    Through advanced machine learning models and comprehensive analysis, we provide users with predictions on the most 
    likely winning team for each fixture, as well as forecasts for the total number of goals to be scored. Our prediction
    models leverage a rich dataset that spans historical match data and head-to-head statistics, extracting insights from 
    a myriad of features. This data-driven approach allows us to deliver accurate and informed predictions, giving enthusiasts
    and sports aficionados a unique tool to enhance their matchday experience.
    
    We bring the data to life through interactive visualizations, showcasing the performance and scorelines of selected teams. 
    Dive into the standings, explore historical trends, and gain a deeper understanding of the game dynamics.
""", style={'fontWeight': 'normal'}),


# Input field for entering games
    html.Div([
        html.Div([
            html.Label('Input teams to predict TotalGoals in a game:', style={'display': 'block', 'text-align': 'center'}),
            dcc.Textarea(
                id='teams-textareas',
                placeholder='Enter games (Team1 Team2), one per line',
                rows=7,
                style={'width': '100%', 'margin': '0 auto', 'display': 'block'}
            ),
            # Add some spacing between the button and the prediction output
            html.Div('', style={'height': '5px'}), 
            html.Button('Predict TotalGoals', id='predict-goals-button', n_clicks=0, style={'background-color': '#0074D9', 'color': 'white', 'border': 'none', 'border-radius': '5px', 'padding': '10px 20px', 'cursor': 'pointer', 'margin': '0 auto', 'display': 'block' }),
            html.Div(id='output-prediction-goals', style={'text-align': 'center'}),
        ], style={'width': '50%'}),

        # Spacing between the two divs (5% width)
        html.Div([], style={'width': '5%', 'display': 'inline-block'}),

        html.Div([
            html.Label('Input teams to predict Winner', style={'display': 'block', 'text-align': 'center'}),
            dcc.Textarea(
                id='games-input',
                placeholder='Enter games (Team1 Team2), one per line',
                rows=7,
                style={'width': '100%', 'margin': '0 auto', 'display': 'block'}
            ),
            html.Div('', style={'height': '5px'}), 
            html.Button('Predict Winner', id='predict-button', n_clicks=0,  
                style={'background-color': '#0074D9', 'color': 'white', 'border': 'none', 'border-radius': '5px', 'padding': '10px 20px', 'cursor': 'pointer', 'margin': '0 auto', 'display': 'block' }),
       
            html.Div(id='prediction-output-container'),
        ], style={'width': '50%', 'margin': '0 auto', 'display': 'block'}),
    ], style={'display': 'flex', 'border': '2px solid #333', 'border-radius': '10px'}),


     # Add spacing of 30px between the input div and the parent div
    html.Div([], style={'height': '50px'}),

    html.H2("Visualiation"),
    # Parent Div for Second and Third Divs
    html.Div([
        # Second Div: Display a sample Plotly plot (45% width)
        html.Div([
        team_dropdown,
        dcc.Graph(id='game-results')
        ], style={'width': '50%', 'display': 'inline-block', 'border': '2px solid #333', 'border-radius': '10px'}),

        # Spacing between the two divs (5% width)
        html.Div([], style={'width': '2%', 'display': 'inline-block'}),

    # Add the table
        html.Div([
        dcc.Dropdown(
        id='country-dropdown',
        options=[{'label': country, 'value': country} for country in unique_countries],
        value='Spain'  # Default selected country
    ),
        dash_table.DataTable(
            id='standings-table',
            page_size=10  # Set the number of rows per page
        )
    ], style={'width': '50%', 'display': 'inline-block', 'border': '2px solid #333', 'border-radius': '10px'})
    ], style={'display': 'flex'}),

         # Add spacing of 30px between the input div and the parent div
    html.Div([], style={'height': '50px'}),

    html.H2("Report"),

    html.Div([
        html.Div(id='magazine-report', style={'color': '#000000', 'font-size': '16px'}),
        # Dummy callback to trigger the report generation
        dcc.Input(id='dummy-input', type='hidden', value='trigger')
    ], style={'background-color': '#f0f0f0', 'width': '100%', 'padding': '20px', 'border-radius': '10px'}),

])

@dash_app.callback(
    Output('magazine-report', 'children'),
    [Input('dummy-input', 'value')]
)
def generate_magazine_report(dummy_input):
    # Filter data for the latest two dates
    latest_two_dates_data = transformed_df.sort_values(by='Date', ascending=False).groupby('Team').head(1)

    # Team with the Highest Continuous Winning Streak
    max_streaks = latest_two_dates_data.groupby('Team')['Continuous Streak'].max()
    teams_highest_streak = max_streaks[max_streaks == max_streaks.max()].index

    # Identify the biggest upset
    max_elo_per_team = latest_two_dates_data.groupby('Team')['Elo Rating'].transform('max')
    biggest_upset = latest_two_dates_data.loc[(latest_two_dates_data['Result'] == 'L') & (latest_two_dates_data['Elo Rating'] == max_elo_per_team)]
    single_biggest_upset = biggest_upset.loc[biggest_upset['Elo Rating'].idxmax()]

    # Game with the most goals
    most_goals_game = latest_two_dates_data.loc[latest_two_dates_data['TotalGoals'].idxmax()]

    # Construct a more engaging Magazine Report
    magazine_report = f"""
    \U0001F525 \U0001F3C6 Breaking News \U0001F3C6 \U0001F525

    Unstoppable Force Alert! \U0001F4A5 Brace yourselves as {', '.join(teams_highest_streak)} crush their rivals,
    securing an incredible {max_streaks.max()} consecutive wins! ⚽💥.

    In a shocking turn of events, the underdogs {single_biggest_upset['Opponent']} faced off against {single_biggest_upset['Team']} and emerged victorious,
    considering {single_biggest_upset['Team']} was on form the past few weeks! 🚨😱

    The game that featured {most_goals_game['Team']} and {most_goals_game['Opponent']} consisted of an {most_goals_game['TotalGoals']} goal thriller!
    Quite Enjoyable and thrilling! ⚽🔥
    """
    return html.Div([dcc.Markdown(magazine_report)])


@dash_app.callback(
    Output('standings-table', 'data'),
    Output('standings-table', 'columns'),
    Input('country-dropdown', 'value')
)

def update_table(selected_country):
    # Filter the DataFrame to include only matches from the selected country
    country_matches = transformed_df[(transformed_df['Country'] == selected_country) & (transformed_df['Date'] >= start_date)]

    # Calculate the points, wins, draws, losses, GS, GC, GD for each team and create the standings DataFrame
    team_stats = {}
    for index, row in country_matches.iterrows():
        team = row['Team']
        result = row['Result']
        goals_scored = row['Goals Scored']
        goals_conceded = row['Goals Conceded']

        if team in team_stats:
            team_stats[team]['Points'] += 3 if result == 'W' else 1 if result == 'D' else 0
            if result == 'W':
                team_stats[team]['Wins'] += 1
            elif result == 'D':
                team_stats[team]['Draws'] += 1
            else:
                team_stats[team]['Losses'] += 1
            team_stats[team]['GS'] += goals_scored
            team_stats[team]['GC'] += goals_conceded
        else:
            points = 3 if result == 'W' else 1 if result == 'D' else 0
            wins = 1 if result == 'W' else 0
            draws = 1 if result == 'D' else 0
            losses = 1 if result == 'L' else 0
            team_stats[team] = {
                'Points': points,
                'Wins': wins,
                'Draws': draws,
                'Losses': losses,
                'GS': goals_scored,
                'GC': goals_conceded,
            }

    for team in team_stats:
        team_stats[team]['GD'] = team_stats[team]['GS'] - team_stats[team]['GC']

    standings_df = pd.DataFrame(list(team_stats.values()), index=team_stats.keys())
    standings_df.index.name = 'Team'
    standings_df.reset_index(inplace=True)
    standings_df = standings_df.sort_values(by='Points', ascending=False).reset_index(drop=True)

    columns = [{'name': col, 'id': col} for col in standings_df.columns]
    data = standings_df.to_dict('records')

    return data, columns

@dash_app.callback(
    Output('game-results', 'figure'),
    Input('team-dropdown', 'value')
)
def update_figure(selected_team):
    # Filter the DataFrame for the selected team's games from the start date to the current date
    selected_team_games = transformed_df[(transformed_df['Date'] >= start_date) & (transformed_df['Team'] == selected_team)]

    # Keep only one match ID per game
    selected_team_games = selected_team_games.drop_duplicates(subset='Match ID')

    # Create the Plotly figure for the selected team
    fig = px.bar(
        selected_team_games,
        x='Date',
        y='Plot Points',
        title=f'{selected_team} Game Results Based on Points',
        labels={'Date': 'Date'},
        color='Points',
        color_continuous_scale=['red', 'blue', 'green'],
        text='Result',
        height=400,
        range_y=[-1, 3]
    )

    # Define a custom hover template
    fig.update_traces(
        hovertemplate="Date: %{x}<br>Points: %{y}<br>Result: %{text}<br>Goals Scored: %{customdata[0]}<br>Goals Conceded: %{customdata[1]}<br>Opponent: %{customdata[2]}"
    )

    # Add Goals Scored, Goals Conceded, and Opponent as custom data to the plot
    fig.data[0].update(customdata=selected_team_games[['Goals Scored', 'Goals Conceded', 'Opponent']])

    return fig
    
# Define the logistic function to calculate win probability
def calculate_win_probability(elo_diff):
    return 1 / (1 + math.exp(-elo_diff / 400))
    
def get_elo_rating(team_name):
    # Try to find the Elo rating for the exact team name
    elo = transformed_df.loc[transformed_df['Team'] == team_name, 'Elo Rating'].values
    if len(elo) > 0:
        return elo[-1]

    # If not found, try to find Elo rating for team names with spaces
    for name in transformed_df['Team']:
        if team_name in name or name in team_name:
            elo = transformed_df.loc[transformed_df['Team'] == name, 'Elo Rating'].values
            if len(elo) > 0:
                return elo[-1]
    return None

def predict_match_outcome(team1, team2):
    # Get Elo ratings for both teams
    elo_team1 = get_elo_rating(team1)
    elo_team2 = get_elo_rating(team2)

    if elo_team1 is not None and elo_team2 is not None:
        win_probability_team1 = calculate_win_probability(elo_team1 - elo_team2)
        win_probability_team2 = 1 - win_probability_team1

        if win_probability_team1 > win_probability_team2:
            return elo_team1, elo_team2, f"{team1} wins", f"{win_probability_team1 * 100:.2f}%", f"{win_probability_team2 * 100:.2f}%"
        elif win_probability_team2 > win_probability_team1:
            return elo_team1, elo_team2, f"{team2} wins", f"{win_probability_team1 * 100:.2f}%", f"{win_probability_team2 * 100:.2f}%"
        else:
            return elo_team1, elo_team2, "It's a draw", f"{win_probability_team1 * 100:.2f}%", f"{win_probability_team2 * 100:.2f}%"
    else:
        return None

# Define callback to update the prediction output for multiple matches
@dash_app.callback(
    Output('prediction-output-container', 'children'),
    Input('predict-button', 'n_clicks'),
    State('games-input', 'value')
)
def predict_matches(n_clicks, games_input):
        if n_clicks is not None and games_input:
            # Split the input into lines to get individual games
            games = games_input.strip().split('\n')
            predictions = []

            for game in games:
                teams = game.strip().split(' ')
                if len(teams) == 2:
                    prediction = predict_match_outcome(teams[0], teams[1])
                    if prediction is not None:
                        elo_team1, elo_team2, outcome, prob_team1, prob_team2 = prediction

                        # Compare probabilities and select the team with the higher probability
                        if prob_team1 >= prob_team2:
                            predictions.append((teams[0], elo_team1, teams[1], elo_team2, outcome, prob_team1, prob_team2))
                        else:
                            predictions.append((teams[1], elo_team2, teams[0], elo_team1, outcome, prob_team2, prob_team1))

            if predictions:
                # Sort the predictions based on the win probability for the selected team (highest to lowest)
                predictions.sort(key=lambda x: x[5], reverse=True)

                # Keep only the top 4 predictions
                top_4_predictions = predictions[:4]

            header_style = {
                    'border-bottom': '2px solid black',  # Add a separating line under the header
                }


            # Display the top 4 predictions
            output = html.Table([
                html.Tr([
                    html.Th('Team 1', style=header_style),  # Apply the header style here
                    html.Th('Team 2', style=header_style),
                    html.Th('Win Probability Team 1 (%)', style=header_style),
                    html.Th('Win Probability Team 2 (%)', style=header_style),
                    html.Th('Predicted Outcome', style=header_style),
                ])
            ] + [
                html.Tr([
                    html.Td(game[0]),
                    html.Td(game[1]),
                    html.Td(game[5]),
                    html.Td(game[6]),
                    html.Td(game[4])
                ])
                for game in top_4_predictions
            ], style={
                    'border': '1px solid black',  # Black border
                    'border-radius': '5px'  # Border radius
              })
            return output
        else:
            return html.Div('No valid game data entered.')

# Define the callback to make predictions
@dash_app.callback(
    Output('output-prediction-goals', 'children'),
    [Input('predict-goals-button', 'n_clicks')],
    [State('teams-textareas', 'value')]
)
def predict_total_goals(n_clicks, teams_textarea):
    if n_clicks > 0:
        # Parse input text area value to get team names
        team_lines = teams_textarea.strip().split('\n')
        teams = [line.split() for line in team_lines]

        # Create a DataFrame with the teams and their statistics
        team_data = pd.DataFrame()
        over_25_predictions = pd.DataFrame(columns=['Team1', 'Team2', 'Prediction_Result'])  # To store Over 2.5 predictions

        for team1, team2 in teams:
            team1_data = transformed_df[transformed_df['Team'] == team1][selected_columns]
            team2_data = transformed_df[transformed_df['Team'] == team2][selected_columns]

            team2_data['Opponent_Strength'] = team2_data['Elo Rating'].mean()
            team2_data.drop(['Elo Rating'], axis=1, inplace=True)
            team2_data.columns = ['Average_Goals_Scored_Opponent', 'Average_Goals_Conceded_Opponent', 'Opponent_Strength']

            input_data = pd.concat([team1_data.mean(), team2_data.mean()]).to_frame().T

            # Adjust predicted_total_goals by subtracting 1.2
            predicted_total_goals = model.predict(input_data)[0] - 1.2

            # Sum of average goals scored and average goals scored by the opponent
            sum_of_goals = input_data['Average_Goals_Scored'].values[0] + input_data['Average_Goals_Scored_Opponent'].values[0]

            # Determine the prediction result
            if sum_of_goals > predicted_total_goals:
                prediction_result = "Over 2.5"
                over_25_predictions = pd.concat([over_25_predictions, pd.DataFrame({'Team1': [team1], 'Team2': [team2], 'Prediction_Result': [prediction_result]})])
            else:
                prediction_result = "Not Certain"

            team_data = pd.concat([team_data, pd.DataFrame({'Team1': [team1], 'Team2': [team2], 'Prediction_Result': [prediction_result]})])

        # Display Over 2.5 predictions if there is at least one, otherwise display all "Not Certain" predictions
        if not over_25_predictions.empty:
            return generate_table(over_25_predictions)
        else:
            return generate_table(team_data)


# Helper function to generate a Dash table from a DataFrame
def generate_table(dataframe, max_rows=10):
    return html.Table(
        # Header
        [html.Tr([html.Th(col) for col in dataframe.columns])] +

        # Body
        [html.Tr([html.Td(dataframe.iloc[i][col]) for col in dataframe.columns]) for i in range(min(len(dataframe), max_rows))]
    )


if __name__ == '__main__':
    # Start the Flask server
    app_flask.run(debug=True)

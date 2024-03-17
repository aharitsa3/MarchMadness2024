from sportsdataverse.mbb import load_mbb_team_boxscore, load_mbb_schedule, mbb_teams
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
import statsmodels.api as sm
import ssl
ssl._create_default_https_context = ssl._create_unverified_context

class MarchMadness:
    def __init__(self):
        self.df = None
        self.teams = None
        
    def load_teams(self, file_name=""):
        teams = None
        if not file_name:
            teams = mbb_teams.espn_mbb_teams(groups=50, return_as_pandas=True)
            teams.to_csv("Data/teams.csv")
        else:
            teams = pd.read_csv(file_name).drop('Unnamed: 0', axis=1)
        
        teams = teams.drop(["team_uid", "team_slug", "team_abbreviation", "team_location", "team_color", 
                            "team_alternate_color", "team_is_active", "team_is_all_star", "team_logos"], axis=1)
        
        self.teams = teams
        return teams
        
    def load_data(self, file_name="", season=0):
        if not file_name:
            if season > 0:
                df = load_mbb_team_boxscore(seasons=[season], return_as_pandas=True)
            else:
                raise "Please enter a valid season"
        df = pd.read_csv(file_name).drop('Unnamed: 0', axis=1)
        dfm = df.drop(["opponent_team_uid", "opponent_team_slug", "opponent_team_location", "opponent_team_name",
                    "opponent_team_abbreviation", "opponent_team_short_display_name",
                    "opponent_team_color", "opponent_team_alternate_color", "opponent_team_logo", "opponent_team_score",
                    "team_location", "team_abbreviation", "team_short_display_name", "team_color", "team_name",
                    "team_alternate_color", "team_logo", "team_slug", "team_uid"], axis=1)

        # dfm = df.join(df, how='cross', rsuffix='_right')
        # dfm = dfm[dfm["game_id"] == dfm["game_id_right"]]
        # dfm = dfm[dfm["team_uid"] != dfm["team_uid_right"]]
        # print(dfm)
        # dfm = dfm[dfm["team_uid"] != dfm["team_uid_right"]]

        # dfm["assists_diff"] = dfm["assists"] - dfm["assists_right"]
        # dfm["blocks_diff"] = dfm["blocks"] - dfm["blocks_right"]
        # dfm["drb_diff"] = dfm["defensive_rebounds"] - dfm["defensive_rebounds_right"]
        # dfm["field_goal_pct_diff"] = dfm["field_goal_pct"] - dfm["field_goal_pct_right"]
        # dfm["ftp_diff"] = dfm["free_throw_pct"] - dfm["free_throw_pct_right"]
        # dfm["orb_diff"] = dfm["offensive_rebounds"] - dfm["offensive_rebounds_right"]
        # dfm["three_pt_fgp_diff"] = dfm["three_point_field_goal_pct"] - dfm["three_point_field_goal_pct_right"]
        # dfm["trb_diff"] = dfm["total_rebounds"] - dfm["total_rebounds_right"]
        # dfm["turnovers_diff"] = dfm["turnovers"] - dfm["turnovers_right"]

        self.df = dfm
        return dfm

    def model(self, iterations=1, extra_drop_columns=[]):
        lex = LabelEncoder()
        ley = LabelEncoder()
        lex.fit(self.df["team_home_away"])
        self.df["team_home_away"] = lex.transform(self.df["team_home_away"])
        y = ley.fit_transform(self.df.loc[:, "team_winner"])
        
        best_esimators = []
        metrics = []
        for i in range(iterations):
            drop_columns = ["team_winner", "game_id", "season", "season_type", "game_date", "game_date_time",
                               "team_display_name", "opponent_team_display_name", "flagrant_fouls", "team_turnovers", 
                               "technical_fouls", "total_technical_fouls"]
            for j in range(i+1):
                drop_columns += extra_drop_columns[j]
                
            x = self.df.drop(drop_columns, axis=1).fillna(0)

            estimator = LogisticRegression()
            param_grid = {
                "penalty": ["l1", "l2", "elasticnet"],
                "C": [1, 5, 10, 50, 100, 200],
                "class_weight": ["balanced", None],
                "solver": ["lbfgs", "saga"]
            }

            xtrain, xtest, ytrain, ytest = train_test_split(x, y, test_size=0.3, random_state=2024)

            clf = GridSearchCV(estimator=estimator, param_grid=param_grid, scoring="f1", cv=3)
            clf.fit(xtrain, ytrain)
            best_esimators.append(clf.best_estimator_)
            for l, c in zip(clf.best_estimator_.feature_names_in_, clf.best_estimator_.coef_[0]):
                print(l, c)
                
            ypreds = clf.predict(xtest)
            s = f1_score(ytest, ypreds)
            metrics.append(s)
            print(best_esimators)
            print(metrics)

if __name__ == "__main__":
    mm = MarchMadness()

    mm_qualy_teams = ["Vermont", "NC State", "Stetson", "Iowa State", "UConn", "Montana St", "Longwood",
                    "Long Beach St", "Charleston", "Western KY", "Oakland", "Saint Peter's", "Akron",
                    "Howard", "Drake", "New Mexico", "Wagner", "Morehead St", "Oregon", "Colgate", "Samford",
                    "McNeese", "Grambling", "South Dakota", "James Madison", "Saint Mary's", "Grand Canyon"]
    
    all_teams = mm.load_teams("Data/teams.csv")    
    # print(all_teams[all_teams["team_name"] == "Gaels"])
    teams_filter = all_teams["team_short_display_name"].apply(lambda x: x in mm_qualy_teams)
    tourny_teams = all_teams[teams_filter].loc[:, "team_id"]
    

    data = mm.load_data("Data/data.csv")
    data_filter = data["team_id"].apply(lambda x: x in tourny_teams.to_list())
    mm.df = data[data_filter]
    data = mm.df
    print(data)
    
    mm.model(iterations=3,
             extra_drop_columns=[[],
                                 ["team_home_away", "field_goals_made", "free_throw_pct", "free_throws_attempted",
                                 "offensive_rebounds", "three_point_field_goal_pct"],
                                 ["team_score", "assists", "blocks", "fast_break_points", "field_goal_pct", "free_throws_made",
                                  "points_in_paint", "steals", "three_point_field_goals_made", "three_point_field_goals_attempted"]])
    """
    with all features...
    best model:
    - LogisticRegression(C=100)
    - f1 = 0.935
    
    condition:
    - any metric with weight < 0.01
    
    need to drop:
    - team_home_away
    - field_goals_made
    - free_throw_pct
    - free_throws_attempted
    - offensive rebounds
    - three_point_field_goal_pct
    
    after 2nd iteration:
    best model:
    - LogisticRegression(C=200)
    - f1 = 0.943
    
    condition:
    - any feature with weight < 0.1
    
    need to drop:
    - team_score
    - assists
    - blocks
    - fast_break_points
    - field_goal_pct
    - free_throws_made
    - points_in_paint
    - steals
    - three_point_field_goals_made
    - three_point_field_goals_attempted
    
    after 3rd iteration:
    best model
    - LogisticRegression(C=50)
    - f1 = 0.923
    """
    
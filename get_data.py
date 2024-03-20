from sportsdataverse.mbb import load_mbb_team_boxscore, load_mbb_schedule, mbb_teams
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, LabelBinarizer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
import pickle
import statsmodels.api as sm
import ssl
ssl._create_default_https_context = ssl._create_unverified_context

import copy

class MarchMadness:
    def __init__(self):
        self.df = pd.DataFrame()
        self.teams = []
        self.all_team_ids = []
        self.tourny_team_ids = []
        self.team_avg_stats = pd.DataFrame()
        self.tourny_team_most_recent_avg = {}
        
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

    def load_team_stats(self, file_name=""):
        if file_name:
            df = pd.read_csv(file_name).drop("Unnamed: 0", axis=1)
            self.team_avg_stats = df
            return df
              
        col_names = ["opponent_team_id", "team_winner", "game_date", "assists", "blocks", 
                     "defensive_rebounds","fast_break_points", "field_goal_pct", "field_goals_made",
                    "field_goals_attempted", "flagrant_fouls", "fouls", "free_throw_pct",
                    "free_throws_made", "free_throws_attempted", "largest_lead",
                    "offensive_rebounds", "points_in_paint", "steals", "team_turnovers",
                    "technical_fouls", "three_point_field_goal_pct", "three_point_field_goals_made",
                    "three_point_field_goals_attempted", "total_rebounds", "total_technical_fouls",
                    "total_turnovers", "turnover_points", "turnovers"]
        all_team_games = pd.DataFrame(columns=["team_id"] + col_names)
        
        if not self.all_team_ids:
            return 0
        
        count = 0
        for team in self.all_team_ids:
            team_games = self.df[self.df["team_id"] == team].sort_values("game_date", ascending=True)
            team_games = team_games.loc[:, col_names].fillna(0).reset_index().drop('index', axis=1)
            if team_games.shape[0] == 0:
                count+=1
            for i, row in team_games.iterrows():
                game_date = row.pop("game_date")
                opponent_team_id = row.pop("opponent_team_id")
                team_winner = row.pop("team_winner")
                if not team_games[team_games["game_date"] < game_date].shape[0]:
                    t = team_games.iloc[0, :]
                    team_data = pd.DataFrame(t.tolist()).transpose()
                    team_data.columns = t.index
                else:
                    team_data = team_games[team_games["game_date"] < game_date]
                    team_data = team_data.reset_index().drop('index', axis=1)
                team_data = team_data.drop(["game_date", "opponent_team_id", "team_winner"], axis=1)
                team_avg_data = team_data.mean()
                team_avg_data["team_winner"] = team_winner
                team_avg_data["game_date"] = game_date
                team_avg_data["team_id"] = team
                team_avg_data["opponent_team_id"] = opponent_team_id
                
                new_index = max(all_team_games.index) + 1 if all_team_games.shape[0] > 0 else 0
                all_team_games.loc[new_index] = team_avg_data
        print(count)
        all_team_games.to_csv("Data/teams_rolling_avg.csv")

        self.team_avg_stats = self.load_team_stats("Data/teams_rolling_avg.csv")
        return self.team_avg_stats
          
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

    def model(self, use_team_ids=False, use_opponent_data=False, iterations=1, extra_drop_columns=[]):
        input_df = copy.deepcopy(self.team_avg_stats)
        y = input_df.pop('team_winner')
        
        if use_opponent_data:
            # insert opponent rolling avg data
            full_opp_team_data = pd.DataFrame()
            for i, row in input_df.iterrows():
                game_date = row["game_date"]
                opp_team_id = row["opponent_team_id"]
                
                opp_team_data = input_df[input_df["team_id"] == opp_team_id]
                # TODO: condition if opponent team does not have average data
                if opp_team_data.shape[0] != 0:
                    opp_team_data_filtered = opp_team_data[opp_team_data["game_date"] < game_date]
                    if opp_team_data_filtered.shape[0] != 0:
                        opp_team_data_avg = opp_team_data_filtered.iloc[-1, :]
                    else:
                        opp_team_data_avg = opp_team_data.iloc[-1, :]
                        
                    opp_team_data_avg = opp_team_data_avg.drop(["team_id", "opponent_team_id", "game_date"])
                    
                    opp_team_avg_recent = pd.DataFrame(opp_team_data_avg.tolist()).transpose()
                    opp_team_avg_recent.columns = [f"opponent_{x}" for x in opp_team_data_avg.index]
                
                    full_opp_team_data = pd.concat([full_opp_team_data, opp_team_avg_recent])
                # if opponent team not present, drop row from input df and output data
                else:
                    input_df = input_df.drop(i)
                    y = y.drop(i)
                    
            input_df = input_df.reset_index().drop('index', axis=1)
            full_opp_team_data = full_opp_team_data.reset_index().drop('index', axis=1)
            # add opponent data to input_df
            input_df = pd.concat([input_df, full_opp_team_data], axis=1)
        
        if use_team_ids:
            team_lb = LabelBinarizer()
            team_id_data = input_df.pop("team_id")
            transformed_team_ids = pd.DataFrame(team_lb.fit_transform(team_id_data))
            transformed_team_ids.columns = [f"team_{x}" for x in transformed_team_ids.columns]
            # save
            with open('Data/team_id_encoder.pkl','wb') as f:
                pickle.dump(team_lb,f)
            
            opponent_team_lb = LabelBinarizer()
            opponent_team_id_data = input_df.pop("opponent_team_id")
            transformed_opp_team_ids = pd.DataFrame(opponent_team_lb.fit_transform(opponent_team_id_data))
            transformed_opp_team_ids.columns = [f"opponent_{x}" for x in transformed_opp_team_ids.columns]
            input_df = pd.concat([transformed_team_ids, transformed_opp_team_ids, input_df],axis=1)
            input_df.columns = [str(x) for x in input_df.columns]
            with open('Data/opponent_team_id_encoder.pkl','wb') as f:
                pickle.dump(opponent_team_lb,f)
        else:
            input_df = input_df.drop(["team_id", "opponent_team_id"], axis=1)
        
        input_df.pop('game_date')
        
        le_y = LabelEncoder()
        transformed_y = le_y.fit_transform(y)
        # save
        with open('Data/label_encoder.pkl','wb') as f:
            pickle.dump(le_y,f)
        
        best_estimators = []
        metrics = []
        pipe = Pipeline(
            [
                ("clf", None)   
            ]
        )
        param_grid = [
            {
                "clf": [LogisticRegression()],
                "clf__penalty": ["l2", "elasticnet"],
                "clf__C": [50, 100, 200, 300, 400],
                "clf__class_weight": ["balanced", None],
                "clf__solver": ["lbfgs", "saga"]
            },
            {
                "clf": [RandomForestClassifier()],
                "clf__n_estimators": [50, 100, 200],
                "clf__class_weight": [None, "balanced_subsample"]
            }
        ]

        xtrain, xtest, ytrain, ytest = train_test_split(input_df, transformed_y, test_size=0.2, random_state=2024)

        clf = GridSearchCV(estimator=pipe, param_grid=param_grid, scoring="f1", cv=3)
        clf.fit(xtrain, ytrain)
        best_model = clf.best_estimator_.steps[0][1]
        best_estimators.append(best_model)
        
        for l, c in zip(best_model.feature_names_in_, best_model.coef_[0]):
            print(l, c)
            
        ypreds = clf.predict(xtest)
        s = f1_score(ytest, ypreds)
        metrics.append(s)
        print(best_estimators)
        print(metrics)
        
        # save
        with open('Data/model.pkl','wb') as f:
            pickle.dump(clf,f)
            print("wrote trained model to 'mode.pkl'")
        
       
        # best_esimators = []
        # metrics = []
        # for i in range(iterations):
        #     drop_columns = ["team_winner", "game_id", "season", "season_type", "game_date", "game_date_time",
        #                        "team_display_name", "opponent_team_display_name", "flagrant_fouls", "team_turnovers", 
        #                        "technical_fouls", "total_technical_fouls"]
        #     for j in range(i+1):
        #         drop_columns += extra_drop_columns[j]
                
        #     x = self.df.drop(drop_columns, axis=1).fillna(0)

    def predict(self, matchups, use_team_ids=False, use_opponent_data=False):
        if not self.tourny_team_most_recent_avg:
            for team in self.tourny_team_ids:
                team_data = self.team_avg_stats[self.team_avg_stats["team_id"] == team].sort_values("game_date", ascending=True)
                team_most_recent_avg = team_data.iloc[-1, :]
                self.tourny_team_most_recent_avg[team] = team_most_recent_avg

        clf = None
        team_id_encoder = None
        opp_team_id_encoder = None
        label_encoder = None

        with open('Data/model.pkl', 'rb') as f:
            clf = pickle.load(f)
        with open('Data/label_encoder.pkl', 'rb') as f:
            label_encoder = pickle.load(f)
            print(label_encoder.classes_)
        if use_team_ids:
            with open('Data/team_id_encoder.pkl', 'rb') as f:
                team_id_encoder = pickle.load(f)
            with open('Data/opponent_team_id_encoder.pkl', 'rb') as f:
                opp_team_id_encoder = pickle.load(f)
             
        for (team1, team2) in matchups:
            team1_d = self.tourny_team_most_recent_avg[team1]
            team1_data = pd.DataFrame(team1_d.tolist()).transpose()
            team1_data.columns = team1_d.index
            team1_data = team1_data.drop(["team_id", "opponent_team_id", "game_date", "team_winner"], axis=1)

            team2_d = self.tourny_team_most_recent_avg[team2]
            team2_data = pd.DataFrame(team2_d.tolist()).transpose()
            team2_data.columns = team2_d.index
            team2_data = team2_data.drop(["team_id", "opponent_team_id", "game_date", "team_winner"], axis=1)
            
            pred1_df = copy.deepcopy(team1_data)
            pred2_df = copy.deepcopy(team2_data)

            if use_opponent_data:
                added_df = team2_data
                added_df.columns = [f"opponent_{x}" for x in added_df.columns]
                pred1_df = pd.concat([pred1_df, added_df], axis=1)
                
                added_df = team1_data
                added_df.columns = [f"opponent_{x}" for x in added_df.columns]
                pred2_df = pd.concat([pred2_df, added_df], axis=1)
            
            if use_team_ids:
                encoded_team_id = team_id_encoder.transform(np.array([team1]))
                encoded_team_id = pd.DataFrame(encoded_team_id)
                encoded_team_id.columns = [f"team_{x}" for x in encoded_team_id.columns]
                
                encoded_opp_team_id = opp_team_id_encoder.transform(np.array([team2]))
                encoded_opp_team_id = pd.DataFrame(encoded_opp_team_id)
                encoded_opp_team_id.columns = [f"opponent_{x}" for x in encoded_opp_team_id.columns]
                pred1_df = pd.concat([encoded_team_id, encoded_opp_team_id, pred1_df], axis=1)
                
                
                encoded_team_id = team_id_encoder.transform(np.array([team2]))
                encoded_team_id = pd.DataFrame(encoded_team_id)
                encoded_team_id.columns = [f"team_{x}" for x in encoded_team_id.columns]
                
                encoded_opp_team_id = opp_team_id_encoder.transform(np.array([team1]))
                encoded_opp_team_id = pd.DataFrame(encoded_opp_team_id)
                encoded_opp_team_id.columns = [f"opponent_{x}" for x in encoded_opp_team_id.columns]
                pred2_df = pd.concat([encoded_team_id, encoded_opp_team_id, pred2_df], axis=1)

            p1_prob = clf.predict_proba(pred1_df)
            max_prob1 = np.max(p1_prob[0])
            p2_prob = clf.predict_proba(pred2_df)
            max_prob2 = np.max(p2_prob[0])
            
            winner = None
            if max_prob1 > max_prob2:
                pred_index = np.argmax(p1_prob)
                result = label_encoder.classes_[pred_index]
                winner = team1 if result else team2
            else:
                pred_index = np.argmax(p2_prob)
                result = label_encoder.classes_[pred_index]
                winner = team2 if result else team1
                

                
            print(p1_prob, p2_prob)
                
                
                
            
                
                

        
        

if __name__ == "__main__":
    mm = MarchMadness()

    mm_qualy_teams = ["Vermont", "NC State", "Stetson", "Iowa State", "UConn", "Montana St", "Longwood",
                    "Long Beach St", "Charleston", "Western KY", "Oakland", "Saint Peter's", "Akron",
                    "Howard", "Drake", "New Mexico", "Wagner", "Morehead St", "Oregon", "Colgate", "Samford",
                    "McNeese", "Grambling", "South Dakota", "James Madison", "Saint Mary's", "Grand Canyon",
                    "Yale", "FAU", "Northwestern", "San Diego St", "UAB", "Auburn", "BYU", "Duquesne",
                    "Illinois", "Washington St", "North Carolina", "Mississippi St",
                    "Michigan St", "Alabama", "Clemson", "Baylor", "Dayton", "Nevada", "Arizona", "Houston",
                    "Nebraska", "Texas A&M", "Wisconsin", "Duke", "Texas Tech", "Kentucky", "Florida", "Marquette",
                    "Purdue", "Utah State", "TCU", "Gonzaga", "Kansas", "South Carolina", "Creighton", "Texas",
                    "Tennessee", "Colorado St", "Virginia", "Colorado", "Boise St"]
    
    all_teams = mm.load_teams("Data/teams.csv")
    # print(all_teams[all_teams["team_name"] == "Aggies"])
    teams_filter = all_teams["team_short_display_name"].apply(lambda x: x in mm_qualy_teams)
    tourny_teams = all_teams[teams_filter]
    tourny_teams = tourny_teams.loc[:, "team_id"]
    mm.tourny_team_ids = tourny_teams.to_list()
    mm.all_team_ids = all_teams.loc[:, "team_id"].tolist()


    data = mm.load_data("Data/data.csv")
    # data_filter = data["team_id"].apply(lambda x: x in mm.tourny_team_ids)
    # mm.df = data[data_filter]
    mm.df = data
    # data = mm.df
    
    team_rolling_avg_stats = mm.load_team_stats("Data/teams_rolling_avg.csv")
    # team_rolling_avg_stats = mm.load_team_stats()
    
    # TODO: add opponent team stats to input df
    # mm.model(use_team_ids=True, use_opponent_data=True)
    
    
    mm.predict(matchups=[(2006, 47)], use_team_ids=True, use_opponent_data=True)
    
    
    # mm.model(iterations=3,
    #          extra_drop_columns=[[],
    #                              ["team_home_away", "field_goals_made", "free_throw_pct", "free_throws_attempted",
    #                              "offensive_rebounds", "three_point_field_goal_pct"],
    #                              ["team_score", "assists", "blocks", "fast_break_points", "field_goal_pct", "free_throws_made",
    #                               "points_in_paint", "steals", "three_point_field_goals_made", "three_point_field_goals_attempted"]])
    """
    with team label binarized, opp team label binarized, team avg, opp team avg
    best model:
    - LogisticRegression(C=50, solver='saga')
    - f1 = 0.68579
    
    with only team avg
    best model:
    - LogisticRegression(C=50)
    - f1 = 0.6217
    
    with team avg and team/opp team label binarized
    best model:
    - LogisticRegression(C=400)
    - f1 = 0.6314
    
    
    
    # below is random notes
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
    
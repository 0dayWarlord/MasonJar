(* MasonJar: Bayesian skill rating for pickleball matches based off Herbrich, Ralf, Tom Minka, and Thore Graepel. "TrueSkill™: a Bayesian skill rating system." Advances in neural information processing systems 19 (2006). *)

(* identifier for a pickleball player. *)

type player_id = string

(* score of a single pickleball game to a target number of points *)

type game_score = {
  player1_points : int;
  player2_points : int;
  target_points  : int;
}

(* completed match result between two players *)

type match_result = {
  player1     : player_id;  (* first player *)
  player2     : player_id;  (* second player *)
  winner      : int option;  (* some(1) = player1 wins, Some(2) = player2 wins, None = draw *)
  final_score : game_score;  (* final completed score *)
}

(* current in-progress match state *)

type current_state = {
  player1 : player_id;  (* First player *)
  player2 : player_id;  (* Second player *)
  score   : game_score;  (* current in-progress score *)
}

(* gaussian skill distribution N(mean, variance) *)

type skill_distribution = {
  mean     : float;
  variance : float;
}

(* prediction of match outcome from the model *)

type prediction = {
  player1_win_probability : float;  (* in [0.0; 1.0] *)
  player2_win_probability : float;  (* in [0.0; 1.0] *)
  likely_winner           : int;  (* 1 = player1, 2 = player2 *)
}

type model

(* default TrueSkill hyperparameters *)

val default_mu0 : float
val default_sigma0 : float
val default_beta : float
val default_tau : float
val default_draw_probability : float
val default_target_points : int

(* build a TrueSkill-style model from historical match results

    - uses Gaussian priors N(mu0, sigma0^2) for new players.
    - uses TrueSkill-style performance variance [beta] and dynamics [tau].
    - processes [match_result]s sequentially, updating player skills online.
    - [score_weight] controls how much the current score influences predictions
      (default: 6.0, meaning score advantage is weighted as 6*beta)
*)

val create_model :
  ?mu0:float ->
  ?sigma0:float ->
  ?beta:float ->
  ?tau:float ->
  ?draw_probability:float ->
  ?score_weight:float ->
  match_result list ->
  model

(* predict win probabilities for an in-progress match.

    - uses TrueSkill-style skill posteriors for both players.
    - combines skill difference with a scoreboard-based adjustment
      derived from the current [score].
*)

val predict :
  model ->
  current_state ->
  prediction

(* incorporate a new completed match result into the model *)

val update_model_with_result :
  model ->
  match_result ->
  model

(* get the full posterior skill distribution N(mean, variance) for a player *)

val skill_of_player :
  model ->
  player_id ->
  skill_distribution option

(* get only the posterior mean skill rating for a player *)

val rating_of_player :
  model ->
  player_id ->
  float option

(* get conservative rating (μ - k*σ) for a player.
    This represents a lower bound on skill with k standard deviations of confidence *)

val conservative_rating_of_player :
  ?k:float ->
  model ->
  player_id ->
  float option

(* get percentile rating for a player. *)

val percentile_rating_of_player :
  ?percentile:float ->
  model ->
  player_id ->
  float option


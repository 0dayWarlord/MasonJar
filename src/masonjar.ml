(* MasonJar is a Bayesian skill based predictive algorithm for pickleball matches based on Herbrich, Ralf, Tom Minka, and Thore Graepel. "TrueSkill™: a Bayesian skill rating system." Advances in neural information processing systems 19 (2006). *)

(* types *)

type player_id = string

type game_score = {
  player1_points : int;
  player2_points : int;
  target_points  : int;
}

type match_result = {
  player1     : player_id;
  player2     : player_id;
  winner      : int option;  (* some(1) = player1 wins, Some(2) = player2 wins, None = draw *)
  final_score : game_score;
}

type current_state = {
  player1 : player_id;
  player2 : player_id;
  score   : game_score;
}

type skill_distribution = {
  mean     : float;
  variance : float;
}

type prediction = {
  player1_win_probability : float;
  player2_win_probability : float;
  likely_winner           : int;  (* 1 = player1, 2 = player2 *)
}

(* player map *)

module PlayerMap : sig
  type 'a t
  val empty : 'a t
  val find_opt : string -> 'a t -> 'a option
  val add : string -> 'a -> 'a t -> 'a t
end = struct
  module M = Map.Make(String)
  type 'a t = 'a M.t
  let empty = M.empty
  let find_opt = M.find_opt
  let add = M.add
end

(* internal Gaussian representation *)

type gaussian = {
  mean     : float;
  variance : float;
}

(* default TrueSkill hyperparameters *)

let default_mu0 = 25.0
let default_sigma0 = default_mu0 /. 3.0
let default_beta = default_sigma0 /. 2.0
let default_tau = default_sigma0 /. 100.0
let default_draw_probability = 0.0
let default_target_points = 11

(* model type *)

type model = {
  skills : gaussian PlayerMap.t;
  mu0 : float;
  sigma0 : float;
  beta : float;
  tau : float;
  draw_probability : float;
  score_weight : float;  (* multiplier for score advantage in predictions (default: 6.0) *)
}

(* helper to create a Gaussian from prior parameters *)

let gaussian_of_prior prior_mean prior_standard_deviation =
  { mean = prior_mean; variance = prior_standard_deviation *. prior_standard_deviation }

(* get player skill, or return prior if not found *)

let skill_of_player_internal model player_id =
  match PlayerMap.find_opt player_id model.skills with
  | Some gaussian_distribution -> gaussian_distribution
  | None -> gaussian_of_prior model.mu0 model.sigma0

(* apply dynamics variance (add τ² to variance) *)

let apply_dynamics dynamics_factor (gaussian_distribution : gaussian) =
  { gaussian_distribution with variance = gaussian_distribution.variance +. (dynamics_factor *. dynamics_factor) }

(* mathematical constants *)

let pi = 4.0 *. atan 1.0

(* standard normal PDF: φ(x) = (1/√(2π)) * exp(-0.5 * x²) *)

let phi x =
  let coefficient = 1.0 /. sqrt (2.0 *. pi) in
  coefficient *. exp (-0.5 *. x *. x)

(* standard normal CDF: Φ(x) using Abramowitz-Stegun approximation *)

let normal_cdf x =
  (* Abramowitz-Stegun approximation for the standard normal CDF *)
  (* Φ(x) = 0.5 * (1 + erf(x/√2)) *)
  (* we use a rational approximation for erf *)
  let a1 =  0.254829592 in
  let a2 = -0.284496736 in
  let a3 =  1.421413741 in
  let a4 = -1.453152027 in
  let a5 =  1.061405429 in
  let p_parameter  =  0.3275911 in
  
  (* sign of x *)

  let sign_multiplier = if x < 0.0 then -1.0 else 1.0 in
  let x = abs_float x in
  
  (* A&S formula 7.1.26 *)

  let t_parameter = 1.0 /. (1.0 +. p_parameter *. x) in
  let y_value = 1.0 -. (((((a5 *. t_parameter +. a4) *. t_parameter) +. a3) *. t_parameter +. a2) *. t_parameter +. a1) *. t_parameter *. exp (-. x *. x) in
  
  0.5 *. (1.0 +. sign_multiplier *. y_value)

(* approximate inverse CDF (quantile function) for standard normal distribution.
    Uses Beasley-Springer-Moro algorithm approximation *)

let inv_normal_cdf probability =
  (* probability must be in (0, 1) *)
  if probability <= 0.0 then neg_infinity
  else if probability >= 1.0 then infinity
  else if probability = 0.5 then 0.0
  else
    (* Beasley-Springer-Moro algorithm *)
    let a0 = 2.50662823884 in
    let a1 = -18.61500062529 in
    let a2 = 41.39119773534 in
    let a3 = -25.44106049637 in
    let b1 = -8.47351093090 in
    let b2 = 23.08336743743 in
    let b3 = -21.06224101826 in
    let b4 = 3.13082909833 in
    let c0 = -2.78718931138 in
    let c1 = -2.29796479134 in
    let c2 = 4.85014127135 in
    let c3 = 2.32121276858 in
    let d1 = 3.54388924762 in
    let d2 = 1.63706781897 in
    
    let centered_probability = probability -. 0.5 in
    if abs_float centered_probability < 0.42 then
      (* central region *)
      let ratio = centered_probability *. centered_probability in
      centered_probability *. (((a3 *. ratio +. a2) *. ratio +. a1) *. ratio +. a0) /.
            ((((b4 *. ratio +. b3) *. ratio +. b2) *. ratio +. b1) *. ratio +. 1.0)
    else
      (* tail region *)
      let tail_probability = if centered_probability < 0.0 then probability else 1.0 -. probability in
      let log_sqrt = sqrt (-. (log tail_probability)) in
      let sign_multiplier = if centered_probability < 0.0 then -1.0 else 1.0 in
      sign_multiplier *. (((c3 *. log_sqrt +. c2) *. log_sqrt +. c1) *. log_sqrt +. c0) /.
              ((d2 *. log_sqrt +. d1) *. log_sqrt +. 1.0)

(* TrueSkill v function for win case (no draws) *)

let v_win normalized_mean_difference =
  (* normalized_mean_difference = μ_d / σ_d *)
  let denominator = normal_cdf normalized_mean_difference in
  if denominator <= 0.0 then
    -. normalized_mean_difference  (* avoid division by zero; very unlikely edge case *)
  else
    phi normalized_mean_difference /. denominator

(* TrueSkill w function for win case (no draws) *)

let w_win normalized_mean_difference =
  let v_function_value = v_win normalized_mean_difference in
  v_function_value *. (v_function_value +. normalized_mean_difference)

(* TrueSkill v function for draw case *)

let v_draw normalized_mean_difference draw_margin_over_sigma =
  (* normalized_mean_difference = μ_d / σ_d, draw_margin_over_sigma = ε / σ_d where ε is the draw margin *)
  let denominator = normal_cdf (draw_margin_over_sigma -. normalized_mean_difference) -. normal_cdf (-.draw_margin_over_sigma -. normalized_mean_difference) in
  if denominator <= 0.0 then
    0.0  (* avoid division by zero *)
  else
    let numerator = phi (draw_margin_over_sigma -. normalized_mean_difference) -. phi (-.draw_margin_over_sigma -. normalized_mean_difference) in
    -. (numerator /. denominator)

(* TrueSkill w function for draw case *)

let w_draw normalized_mean_difference draw_margin_over_sigma =
  let v_function_value = v_draw normalized_mean_difference draw_margin_over_sigma in
  let denominator = normal_cdf (draw_margin_over_sigma -. normalized_mean_difference) -. normal_cdf (-.draw_margin_over_sigma -. normalized_mean_difference) in
  if denominator <= 0.0 then
    0.0
  else
    let numerator = (draw_margin_over_sigma -. normalized_mean_difference) *. phi (draw_margin_over_sigma -. normalized_mean_difference) +. (draw_margin_over_sigma +. normalized_mean_difference) *. phi (-.draw_margin_over_sigma -. normalized_mean_difference) in
    v_function_value *. v_function_value +. (numerator /. denominator)

(* update two players after a match (winner vs loser, no draws) *)

let update_two_player_win_no_draw
    ~beta
    (winner_skill : gaussian)
    (loser_skill  : gaussian) : gaussian * gaussian =
  let winner_variance = winner_skill.variance in
  let loser_variance = loser_skill.variance in
  let combined_standard_deviation = sqrt (2.0 *. beta *. beta +. winner_variance +. loser_variance) in
  let normalized_mean_difference = (winner_skill.mean -. loser_skill.mean) /. combined_standard_deviation in
  let v_function_value = v_win normalized_mean_difference in
  let w_function_value = w_win normalized_mean_difference in
  let winner_variance_over_combined = winner_variance /. combined_standard_deviation in
  let loser_variance_over_combined = loser_variance /. combined_standard_deviation in
  let new_mean_winner  = winner_skill.mean +. winner_variance_over_combined *. v_function_value in
  let new_mean_loser   = loser_skill.mean -. loser_variance_over_combined *. v_function_value in
  let new_variance_winner =
    winner_variance *. (1.0 -. (winner_variance /. (combined_standard_deviation *. combined_standard_deviation)) *. w_function_value)
  in
  let new_variance_loser =
    loser_variance *. (1.0 -. (loser_variance /. (combined_standard_deviation *. combined_standard_deviation)) *. w_function_value)
  in
  ({ mean = new_mean_winner; variance = new_variance_winner },
   { mean = new_mean_loser;  variance = new_variance_loser })

(* update two players after a draw *)

let update_two_player_draw
    ~beta
    ~draw_probability
    (player1_skill : gaussian)
    (player2_skill : gaussian) : gaussian * gaussian =
  let player1_variance = player1_skill.variance in
  let player2_variance = player2_skill.variance in
  let combined_standard_deviation = sqrt (2.0 *. beta *. beta +. player1_variance +. player2_variance) in
  let normalized_mean_difference = (player1_skill.mean -. player2_skill.mean) /. combined_standard_deviation in
  (* draw margin: ε = β * sqrt(2) * Φ^(-1)((1 + draw_probability) / 2) *)
  (* for draw_probability = 0.0, draw_margin = 0, which gives symmetric updates *)
  let draw_margin = if draw_probability > 0.0 then
      let quantile = (1.0 +. draw_probability) /. 2.0 in
      beta *. sqrt 2.0 *. inv_normal_cdf quantile
    else
      0.0
  in
  let draw_margin_over_combined = draw_margin /. combined_standard_deviation in
  let v_function_value = v_draw normalized_mean_difference draw_margin_over_combined in
  let w_function_value = w_draw normalized_mean_difference draw_margin_over_combined in
  let player1_variance_over_combined = player1_variance /. combined_standard_deviation in
  let player2_variance_over_combined = player2_variance /. combined_standard_deviation in
  let new_mean_player1 = player1_skill.mean +. player1_variance_over_combined *. v_function_value in
  let new_mean_player2 = player2_skill.mean -. player2_variance_over_combined *. v_function_value in
  let new_variance_player1 = player1_variance *. (1.0 -. (player1_variance /. (combined_standard_deviation *. combined_standard_deviation)) *. w_function_value) in
  let new_variance_player2 = player2_variance *. (1.0 -. (player2_variance /. (combined_standard_deviation *. combined_standard_deviation)) *. w_function_value) in
  ({ mean = new_mean_player1; variance = new_variance_player1 },
   { mean = new_mean_player2; variance = new_variance_player2 })

(* score advantage feature: normalized point difference *)

let score_advantage (score : game_score) : float =
  let point_difference = float_of_int (score.player1_points - score.player2_points) in
  let target_points  = float_of_int (max 1 score.target_points) in
  point_difference /. target_points  (* roughly in [-1, 1] *)

(* public API *)

let create_model ?mu0 ?sigma0 ?beta ?tau ?draw_probability ?score_weight (matches : match_result list) =
  let mu0 = Option.value mu0 ~default:default_mu0 in
  let sigma0 = Option.value sigma0 ~default:default_sigma0 in
  let beta = Option.value beta ~default:default_beta in
  let tau = Option.value tau ~default:default_tau in
  let draw_probability =
    Option.value draw_probability ~default:default_draw_probability
  in
  let score_weight = Option.value score_weight ~default:6.0 in
  let initial = {
    skills = PlayerMap.empty;
    mu0; sigma0; beta; tau; draw_probability; score_weight;
  } in
  List.fold_left
    (fun model (match_result : match_result) ->
      (* get player skills BEFORE dynamics *)
      let player1_skill_before = skill_of_player_internal model match_result.player1 in
      let player2_skill_before = skill_of_player_internal model match_result.player2 in
      
      (* apply dynamics to players *)
      let player1_skill_after_dynamics = apply_dynamics model.tau player1_skill_before in
      let player2_skill_after_dynamics = apply_dynamics model.tau player2_skill_before in
      
      match match_result.winner with
      | Some winner_number ->
        (* winner/loser case *)
        let winner_is_player1 = (winner_number = 1) in
        let winner_skill, loser_skill =
          if winner_is_player1 then player1_skill_after_dynamics, player2_skill_after_dynamics
          else player2_skill_after_dynamics, player1_skill_after_dynamics
        in
        let updated_winner_skill, updated_loser_skill =
          update_two_player_win_no_draw ~beta:model.beta winner_skill loser_skill
        in
        
        (* update skills map *)

        let updated_player1_skill, updated_player2_skill =
          if winner_is_player1 then updated_winner_skill, updated_loser_skill
          else updated_loser_skill, updated_winner_skill
        in
        let skills =
          PlayerMap.add match_result.player1 updated_player1_skill
            (PlayerMap.add match_result.player2 updated_player2_skill model.skills)
        in
        { model with skills }
      | None ->
        (* draw case *)
        let updated_player1_skill, updated_player2_skill =
          update_two_player_draw ~beta:model.beta ~draw_probability:model.draw_probability
            player1_skill_after_dynamics player2_skill_after_dynamics
        in
        
        (* more updating *)

        let skills =
          PlayerMap.add match_result.player1 updated_player1_skill
            (PlayerMap.add match_result.player2 updated_player2_skill model.skills)
        in
        { model with skills })
    initial
    matches

let update_model_with_result model (match_result : match_result) =
  (* get player skills BEFORE dynamics *)
  let player1_skill_before = skill_of_player_internal model match_result.player1 in
  let player2_skill_before = skill_of_player_internal model match_result.player2 in
  
  (* apply dynamics to players *)

  let player1_skill_after_dynamics = apply_dynamics model.tau player1_skill_before in
  let player2_skill_after_dynamics = apply_dynamics model.tau player2_skill_before in
  
  match match_result.winner with
  | Some winner_number ->
    (* winner/loser case *)
    let winner_is_player1 = (winner_number = 1) in
    let winner_skill, loser_skill =
      if winner_is_player1 then player1_skill_after_dynamics, player2_skill_after_dynamics
      else player2_skill_after_dynamics, player1_skill_after_dynamics
    in
    let updated_winner_skill, updated_loser_skill =
      update_two_player_win_no_draw ~beta:model.beta winner_skill loser_skill
    in
    
    (* even more updating *)

    let updated_player1_skill, updated_player2_skill =
      if winner_is_player1 then updated_winner_skill, updated_loser_skill
      else updated_loser_skill, updated_winner_skill
    in
    let skills =
      PlayerMap.add match_result.player1 updated_player1_skill
        (PlayerMap.add match_result.player2 updated_player2_skill model.skills)
    in
    { model with skills }
  | None ->
    (* draw case *)
    let updated_player1_skill, updated_player2_skill =
      update_two_player_draw ~beta:model.beta ~draw_probability:model.draw_probability
        player1_skill_after_dynamics player2_skill_after_dynamics
    in
    
    (* the most updating *)

    let skills =
      PlayerMap.add match_result.player1 updated_player1_skill
        (PlayerMap.add match_result.player2 updated_player2_skill model.skills)
    in
    { model with skills }

let predict model (state : current_state) : prediction =
  (* get player skills *)
  let player1_skill = skill_of_player_internal model state.player1 in
  let player2_skill = skill_of_player_internal model state.player2 in
  
  let base_mean_difference = player1_skill.mean -. player2_skill.mean in
  let combined_variance = 2.0 *. model.beta *. model.beta +. player1_skill.variance +. player2_skill.variance in
  (* ensure variance is non-negative to avoid NaN from sqrt *)
  let combined_variance = max combined_variance 0.0 in
  let standard_deviation = sqrt combined_variance in

  let score_advantage_value = score_advantage state.score in
  (* use configurable score weight, normalized by beta to keep it probabilistically grounded *)
  (* the score weight is a multiplier on beta, so lambda_score = score_weight * beta *)
  (* this ensures the score adjustment scales appropriately with the performance variance *)
  let lambda_score = model.score_weight *. model.beta in
  let adjusted_mean_difference = base_mean_difference +. lambda_score *. score_advantage_value in

  (* avoid division by zero *)

  let normalized_mean_difference = if standard_deviation > 0.0 then adjusted_mean_difference /. standard_deviation else 0.0 in
  let player1_win_probability = normal_cdf normalized_mean_difference in
  let player1_win_probability =
    if player1_win_probability < 0.0 then 0.0
    else if player1_win_probability > 1.0 then 1.0
    else player1_win_probability
  in
  let player2_win_probability = 1.0 -. player1_win_probability in
  let likely_winner = if player1_win_probability >= player2_win_probability then 1 else 2 in
  {
    player1_win_probability;
    player2_win_probability;
    likely_winner;
  }

let skill_of_player model player_id : skill_distribution option =
  let gaussian_distribution = skill_of_player_internal model player_id in
  Some { mean = gaussian_distribution.mean; variance = gaussian_distribution.variance }

let rating_of_player model player_id =
  match skill_of_player model player_id with
  | None -> None
  | Some skill_distribution -> Some skill_distribution.mean

(* get conservative rating (μ - k*σ) for a player.
    This represents a lower bound on skill with k standard deviations of confidence. *)

let conservative_rating_of_player ?(k=3.0) model player_id =
  match skill_of_player model player_id with
  | None -> None
  | Some skill_distribution ->
    let standard_deviation = sqrt skill_distribution.variance in
    Some (skill_distribution.mean -. k *. standard_deviation)

(* get percentile rating for a player.
    Returns the skill value at the given percentile (0.0 to 1.0). *)

let percentile_rating_of_player ?(percentile=0.5) model player_id =
  match skill_of_player model player_id with
  | None -> None
  | Some skill_distribution ->
    if percentile < 0.0 || percentile > 1.0 then
      None
    else

      (* use inverse CDF for normal distribution *)
      (* for percentile probability, we want μ + σ * Φ^(-1)(probability) *)
      
      let standard_deviation = sqrt skill_distribution.variance in
      let quantile = inv_normal_cdf percentile in
      Some (skill_distribution.mean +. quantile *. standard_deviation)


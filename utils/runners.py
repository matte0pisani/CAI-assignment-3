import shutil
from collections import defaultdict
from itertools import permutations
from math import factorial, prod, sqrt, ceil
from pathlib import Path
from typing import Tuple

import pandas as pd
from geniusweb.profile.utilityspace.LinearAdditiveUtilitySpace import (
    LinearAdditiveUtilitySpace,
)
from geniusweb.profileconnection.ProfileConnectionFactory import (
    ProfileConnectionFactory,
)
from geniusweb.protocol.NegoSettings import NegoSettings
from geniusweb.protocol.session.saop.SAOPState import SAOPState
from geniusweb.simplerunner.ClassPathConnectionFactory import ClassPathConnectionFactory
from geniusweb.simplerunner.NegoRunner import StdOutReporter
from geniusweb.simplerunner.Runner import Runner
from pyson.ObjectMapper import ObjectMapper
from uri.uri import URI

from utils.ask_proceed import ask_proceed

from geniusweb.bidspace.AllBidsList import AllBidsList


def run_session(settings) -> Tuple[dict, dict]:
    agents = settings["agents"]
    profiles = settings["profiles"]
    deadline_time_ms = settings["deadline_time_ms"]

    # quick and dirty checks
    assert isinstance(agents, list) and len(agents) == 2
    assert isinstance(profiles, list) and len(profiles) == 2
    assert isinstance(deadline_time_ms, int) and deadline_time_ms > 0
    assert all(["class" in agent for agent in agents])

    for agent in agents:
        if "parameters" in agent:
            if "storage_dir" in agent["parameters"]:
                storage_dir = Path(agent["parameters"]["storage_dir"])
                if not storage_dir.exists():
                    storage_dir.mkdir(parents=True)

    # file path to uri
    profiles_uri = [f"file:{x}" for x in profiles]

    # create full settings dictionary that geniusweb requires
    settings_full = {
        "SAOPSettings": {
            "participants": [
                {
                    "TeamInfo": {
                        "parties": [
                            {
                                "party": {
                                    "partyref": f"pythonpath:{agents[0]['class']}",
                                    "parameters": agents[0]["parameters"]
                                    if "parameters" in agents[0]
                                    else {},
                                },
                                "profile": profiles_uri[0],
                            }
                        ]
                    }
                },
                {
                    "TeamInfo": {
                        "parties": [
                            {
                                "party": {
                                    "partyref": f"pythonpath:{agents[1]['class']}",
                                    "parameters": agents[1]["parameters"]
                                    if "parameters" in agents[1]
                                    else {},
                                },
                                "profile": profiles_uri[1],
                            }
                        ]
                    }
                },
            ],
            # "deadline": {"DeadlineRounds": {"rounds": rounds, "durationms": 60000}},
            "deadline": {"DeadlineTime": {"durationms": deadline_time_ms}},
        }
    }

    # parse settings dict to settings object
    settings_obj = ObjectMapper().parse(settings_full, NegoSettings)

    # create the negotiation session runner object
    runner = Runner(settings_obj, ClassPathConnectionFactory(), StdOutReporter(), 0)

    # run the negotiation session
    runner.run()

    # get results from the session in class format and dict format
    results_class: SAOPState = runner.getProtocol().getState()
    results_dict: dict = ObjectMapper().toJson(results_class)["SAOPState"]

    # add utilities to the results and create a summary
    results_trace, results_summary = process_results(results_class, results_dict)

    return results_trace, results_summary


def run_tournament(tournament_settings: dict) -> Tuple[list, list]:
    # create agent permutations, ensures that every agent plays against every other agent on both sides of a profile set.
    agents = tournament_settings["agents"]
    profile_sets = tournament_settings["profile_sets"]
    deadline_time_ms = tournament_settings["deadline_time_ms"]

    num_sessions = (factorial(len(agents)) // factorial(len(agents) - 2)) * len(
        profile_sets
    )
    if num_sessions > 100:
        message = (
            f"WARNING: this would run {num_sessions} negotiation sessions. Proceed?"
        )
        if not ask_proceed(message):
            print("Exiting script")
            exit()

    tournament_results = []
    tournament_steps = []
    for profiles in profile_sets:
        # quick an dirty check
        assert isinstance(profiles, list) and len(profiles) == 2
        for agent_duo in permutations(agents, 2):
            # create session settings dict
            settings = {
                "agents": list(agent_duo),
                "profiles": profiles,
                "deadline_time_ms": deadline_time_ms,
            }

            # run a single negotiation session
            _, session_results_summary = run_session(settings)

            # assemble results
            tournament_steps.append(settings)
            tournament_results.append(session_results_summary)

    tournament_results_summary = process_tournament_results(tournament_results)

    return tournament_steps, tournament_results, tournament_results_summary


def process_results(results_class: SAOPState, results_dict: dict):
    print("computing metrics...")

    # dict to translate geniusweb agent reference to Python class name
    agent_translate = {
        k: v["party"]["partyref"].split(".")[-1]
        for k, v in results_dict["partyprofiles"].items()
    }

    # initialize some values
    results_summary = {"num_offers": 0, 
                       "%_fortunate": 0,
                       "%_concession": 0,
                       "%_unfortunate": 0,
                       "%_selfish": 0,
                       "%_nice": 0,
                       "%_silent": 0,
                       "sensibility_to_preferences": 0,
                       "sensibility_to_behaviour": 0}
    num_offers_self = 0

    # check if there are any actions (could have crashed)
    if results_dict["actions"]:
        # obtain utility functions; you actually get the profile for each player
        utility_funcs = {
            k: get_utility_function(v["profile"])
            for k, v in results_dict["partyprofiles"].items()
        }

        # compute the Nash point, the KS point and the POF
        # get all possible bids in the current domain, starting from one of the
        # profile objects
        all_bids = AllBidsList(list(utility_funcs.values())[0].getDomain())
        np, ks, pof = compute_pareto_frontier(all_bids, list(utility_funcs.values())[0], 
                                              list(utility_funcs.values())[1])
        
        # I could remove some bids, otherwise it takes too much time to compute 
        # sensibility to preferences!
        dummy = []
        reduction_factor = 0.5
        for i in range(int(ceil(all_bids.size()*reduction_factor))):
            dummy.append(all_bids.get(i))
        all_bids = dummy

        # iterate both action classes and dict entries
        actions_iter = zip(results_class.getActions(), results_dict["actions"])

        # this is to compare the players current bid with the previous one
        old_bid = None

        for action_class, action_dict in actions_iter:
            if "Offer" in action_dict:
                offer = action_dict["Offer"]
            elif "Accept" in action_dict:
                offer = action_dict["Accept"]
            else:
                continue

            # add bid utility of both agents if bid is not None
            bid = action_class.getBid()
            if bid is None:
                raise ValueError(
                    f"Found `None` value in sequence of actions: {action_class}"
                )
            else:
                offer["utilities"] = {
                    k: float(v.getUtility(bid)) for k, v in utility_funcs.items()
                }

            # if we have a move of our agent, we compute the kind of move (selfish, concession, ...)
            # and we accumulate the result for the sensibility over opponent preference metric
            agent_id_string = "agents_group52_agent_group52_agent_Group52Agent" # from debugging code
            if bid is not None and agent_id_string in offer["actor"]:

                # I retrieve if my agent is the 1st or 2nd in the current negotiation and 
                # get his profile and the adversary's
                idx = (int(offer["actor"][-1]) - 1) % 2 # the % is needed when multiple executions
                prof_self = list(utility_funcs.values())[idx]
                prof_other = list(utility_funcs.values())[1-idx]

                # ***STUFF TO COMPUTE SENSIBILITY TO BEHAVIOUR***
                
                # I first compute the category of a move, considering the relationship with the previous
                # one; then I increase the right counter
                if old_bid is not None:
                    classe = compute_step_class(bid, old_bid, prof_self, prof_other)
                    if classe != "other":
                        results_summary["%_" + classe] += 1
                old_bid = bid
                
                # ***STUFF TO COMPUTE SENSIBILITY TO PREFERENCES***

                # print(all_bids.size())
                # I take all the bids which have same self utility as the current bid
                # since there are (apparently) no 2 bids with the same exact utility, in order to compute an approximation of the 
                # sensibility over preferences metric we also consider the other bids which are very very close in self utility (+- 0.01)
                tolerance = 0.03
                isocurve_bids = list(filter(lambda x: float(prof_self.getUtility(bid))-tolerance < float(prof_self.getUtility(x)) < float(prof_self.getUtility(bid))+tolerance, all_bids))

                if(len(isocurve_bids) != 0):
                    # if(len(isocurve_bids) > 1):
                        # print(len(isocurve_bids))
                    # I select among the isocurve bids the one which maximizes the opponent's utility
                    best_for_opp = max(isocurve_bids, key=lambda b: prof_other.getUtility(b))
                    # I accumulate the deltas in this variable; in the end I will divide it for the number of
                    # self's bids
                    difference = prof_other.getUtility(best_for_opp)-prof_other.getUtility(bid)
                    # print(difference)
                    results_summary["sensibility_to_preferences"] += float(difference)
                
                num_offers_self += 1

            results_summary["num_offers"] += 1

        # gather a summary of results
        if "Accept" in action_dict:
            utilities_final = list(offer["utilities"].values())
            result = "agreement"
        else:
            utilities_final = [0, 0]
            result = "failed"
    else:
        utilities_final = [0, 0]
        result = "ERROR"

    # ***FINAL RESULTS***

    for i, actor in enumerate(results_dict["connections"]):
        position = actor.split("_")[-1]
        results_summary[f"agent_{position}"] = agent_translate[actor]
        results_summary[f"utility_{position}"] = utilities_final[i]

    results_summary["nash_product"] = prod(utilities_final)
    results_summary["social_welfare"] = sum(utilities_final)
    results_summary["distance_nash_point"] = dist(utilities_final[0], utilities_final[1], np)
    results_summary["distance_kalai_smorodisnky_point"] = dist(utilities_final[0], utilities_final[1], ks)
    minimum_distance_point = min(pof, key=lambda p: dist(utilities_final[0], utilities_final[1], p))
    results_summary["distance_pareto_optimal_frontier"] = dist(utilities_final[0], utilities_final[1], minimum_distance_point)
    
    if num_offers_self > 0:
        results_summary["sensibility_to_preferences"] = results_summary["sensibility_to_preferences"] / num_offers_self
        results_summary["%_fortunate"] = results_summary["%_fortunate"] / num_offers_self
        results_summary["%_selfish"] = results_summary["%_selfish"] / num_offers_self
        results_summary["%_concession"] = results_summary["%_concession"] / num_offers_self
        results_summary["%_unfortunate"] = results_summary["%_unfortunate"] / num_offers_self
        results_summary["%_nice"] = results_summary["%_nice"] / num_offers_self
        results_summary["%_silent"] = results_summary["%_silent"] / num_offers_self
        if (results_summary["%_unfortunate"] + results_summary["%_selfish"] + results_summary["%_silent"]) != 0:
            results_summary["sensibility_to_behaviour"] = (results_summary["%_fortunate"] + results_summary["%_concession"] + results_summary["%_nice"]) / \
                                                            (results_summary["%_unfortunate"] + results_summary["%_selfish"] + results_summary["%_silent"])
        else:
            results_summary["sensibility_to_behaviour"] = "inf"

    results_summary["result"] = result

    print("done!")
    return results_dict, results_summary


def get_utility_function(profile_uri) -> LinearAdditiveUtilitySpace:
    profile_connection = ProfileConnectionFactory.create(
        URI(profile_uri), StdOutReporter()
    )
    profile = profile_connection.getProfile()
    assert isinstance(profile, LinearAdditiveUtilitySpace)

    return profile


def process_tournament_results(tournament_results):
    agent_result_raw = defaultdict(lambda: defaultdict(list))
    tournament_results_summary = defaultdict(lambda: defaultdict(int))
    for session_results in tournament_results:
        agents = {k: v for k, v in session_results.items() if k.startswith("agent")}
        for agent_id, agent_class in agents.items():
            agent_result_raw[agent_class]["utility"].append(
                session_results[f"utility_{agent_id.split('_')[1]}"]
            )
            agent_result_raw[agent_class]["nash_product"].append(
                session_results["nash_product"]
            )
            agent_result_raw[agent_class]["social_welfare"].append(
                session_results["social_welfare"]
            )
            if "num_offers" in session_results:
                agent_result_raw[agent_class]["num_offers"].append(
                    session_results["num_offers"]
                )
            tournament_results_summary[agent_class][session_results["result"]] += 1

    for agent, stats in agent_result_raw.items():
        num_session = len(stats["utility"])
        for desc, stat in stats.items():
            stat_average = sum(stat) / num_session
            tournament_results_summary[agent][f"avg_{desc}"] = stat_average
        tournament_results_summary[agent]["count"] = num_session

    column_order = [
        "avg_utility",
        "avg_nash_product",
        "avg_social_welfare",
        "avg_num_offers",
        "count",
        "agreement",
        "failed",
        "ERROR",
    ]
    column_type = {
        "count": int,
        "agreement": int,
        "failed": int,
        "ERROR": int,
    }

    # results dictionary to dataframe
    tournament_results_summary = pd.DataFrame(tournament_results_summary).T

    # clean data and types
    tournament_results_summary = tournament_results_summary.fillna(0)
    for column in column_order:
        if column not in tournament_results_summary:
            tournament_results_summary[column] = 0
    tournament_results_summary = tournament_results_summary.astype(column_type)

    # structure dataframe
    tournament_results_summary.sort_values("avg_utility", ascending=False, inplace=True)
    tournament_results_summary = tournament_results_summary[column_order]

    return tournament_results_summary

def compute_pareto_frontier(all_bids, profile_0, profile_1):    
    """
        Computes the POF, the Nash point and the Kalai-S... point.
        Returns the three as a triple; for NP and KS, returns the bid and the respective 
        utilities for both players; for the POF it returns a list of (bid, u1, u2) values.
    """

    # initialize relevant points and pof list
    pof = []
    nash = None
    ks = None

    for i in range(all_bids.size()):
        bid = all_bids.get(i)
        u0 = float(profile_0.getUtility(bid))
        u1 = float(profile_1.getUtility(bid))

        # the point is on the pof if there in no other pof point wich dominates him.
        # So i check that and if it's verified i add it to the list. Also for each
        # point I check if it dominates any point in the pof list: if yes, I have to
        # remove it
        if len(pof) == 0:
            pof.append((bid, u0, u1))
        else:
            dominating_points = list(filter(lambda p: (p[1] > u0 and p[2] >= u1) or 
                                            (p[1] >= u0 and p[2] > u1), pof))
            dominated_points = list(filter(lambda p: (u0 > p[1] and u1 >= p[2]) or 
                                            (u0 >= p[1] and u1 > p[2]), pof))

            pof = list(set(pof) - set(dominated_points))
            if len(dominating_points) == 0:
                pof.append((bid, u0, u1))
            

        # if it's the first point or if the current point is has a utilities product bigger
        # than the one with the highest product up to now, it's the new nash product
        if nash is None or u0*u1 > nash[1]*nash[2]:
            nash = (bid, u0, u1)

        # similarly, if I have no KS (first point) or if the ratio between the 2 current utilities
        # is closer to 1 than that of the previous "best KS approximation", we have a new best approximation
        # we exclude cases where u1 is 0; in that case, 0,0 would be a KS, but I think it's very unlikely that
        # such extreme case happens
        if u1 != 0 and (ks is None or abs(1-(u0/u1)) < abs(1-(ks[1]/ks[2]))):
            ks = (bid, u0, u1)

    return nash, ks, pof

def dist(u0, u1, point):
    return sqrt((u0 - point[1]) ** 2 + (u1 - point[2]) ** 2)

def compute_step_class(bid, old_bid, prof_self, prof_other):
    u_s = prof_self.getUtility(bid)
    u_o = prof_other.getUtility(bid)
    u_s_old = prof_self.getUtility(old_bid)
    u_o_old = prof_other.getUtility(old_bid)
    tolerance = 0.05

    if u_s > u_s_old and u_o > u_o_old:
        return "fortunate"
    if u_s > u_s_old and u_o <= u_o_old:
        return "selfish"
    if u_s < u_s_old and u_o >= u_o_old:
        return "concession"
    if u_s <= u_s_old and u_o < u_o_old:
        return "unfortunate"
    if (float(u_s_old) - tolerance <= float(u_s) <= float(u_s_old)) and u_o > u_o_old:
        return "nice"
    if (float(u_s_old) - tolerance <= float(u_s) <= float(u_s_old) + tolerance) and (float(u_o_old) - tolerance <= float(u_o) <= float(u_o_old) + tolerance):
        return "silent"
    
    return "other"
    
    
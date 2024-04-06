import logging
from random import randint
from time import time
from typing import cast

from math import ceil

from geniusweb.actions.Accept import Accept
from geniusweb.actions.Action import Action
from geniusweb.actions.Offer import Offer
from geniusweb.actions.PartyId import PartyId
from geniusweb.bidspace.AllBidsList import AllBidsList
from geniusweb.inform.ActionDone import ActionDone
from geniusweb.inform.Finished import Finished
from geniusweb.inform.Inform import Inform
from geniusweb.inform.Settings import Settings
from geniusweb.inform.YourTurn import YourTurn
from geniusweb.issuevalue.Bid import Bid
from geniusweb.issuevalue.Domain import Domain
from geniusweb.party.Capabilities import Capabilities
from geniusweb.party.DefaultParty import DefaultParty
from geniusweb.profile.utilityspace.LinearAdditiveUtilitySpace import (
    LinearAdditiveUtilitySpace,
)
from geniusweb.profileconnection.ProfileConnectionFactory import (
    ProfileConnectionFactory,
)
from geniusweb.progress.ProgressTime import ProgressTime
from geniusweb.references.Parameters import Parameters
from tudelft_utilities_logging.ReportToLogger import ReportToLogger

from .utils.opponent_model import OpponentModel


class Group52Agent(DefaultParty):
    """
    A Python geniusweb agent implemented by Group 52 of the CAI course in TU Delft.
    """

    def __init__(self):
        super().__init__()
        self.logger: ReportToLogger = self.getReporter()

        self.domain: Domain = None
        self.parameters: Parameters = None
        self.profile: LinearAdditiveUtilitySpace = None
        self.progress: ProgressTime = None
        self.me: PartyId = None
        self.other: str = None
        self.settings: Settings = None
        self.storage_dir: str = None

        self.last_received_bid: Bid = None
        self.opponent_model: OpponentModel = None
        self.logger.log(logging.INFO, "party is initialized")

        # the thresholds distinguish the exploration phase from the "real" negotiation one and
        # the final phase (which is in turn divided in 2 by the last variable)
        self.exploration_thresh = 0.25
        self.last_moments_thresh = 0.9
        self.very_last_moments_thresh = 0.95

        # this represents the level from which we consider bids in the first phase; we take only
        # ... % best, according to this parameter
        self.best_bids_percent = 1/100
        self.best_bids = None

        # similar as above, but we consider a lower threshold in the second phase
        self.acceptable_bids_percent = 1/5
        self.acceptable_bids = None

        # tolerance is used when selecting bid in the second phase to give some variance.
        # we don't simply pick the bid with the highest score, but we choose among the ones
        # which have score closer to that, randomly
        self.tolerance = 0.075

        # if we receive a bid with this utility value for us or above in the exploration phase,
        # we accept it
        self.exploration_accept_value = 0.9

        # stores the best opponent's bid from our utility pov
        self.best_received_bid = None

        # these are the initial and final values for the weighted sum of utilities we have to do in the
        # "real" negotiation phase
        self.alpha = 0.8
        self.alpha_max = self.alpha
        self.alpha_min = 0.5

        # opponent tracker: these variables are used to track the behaviour of the opponent and 
        # act consequently
        self.macro_win = 100 # this and the next are the sizes of the 2 moving average windows we use to
        self.micro_win = 10 # analyze the opponent's behaviour
        self.opp_bid_tracker = [] # this to track the opponent's bids
        self.macro_opp_move_type = None # this to represent the "trend" in the opponent behaviour (see later)
        self.micro_opp_move_type = None
        self.time_old = self.exploration_thresh # this is used to update alpha according to the opponent's behaviour

    def notifyChange(self, data: Inform):
        """
        This is the entry point of all interaction with your agent after is has been initialised.
        How to handle the received data is based on its class type.

        Args:
            data (Inform): Contains either a request for action or information.
        """

        # a Settings message is the first message that will be send to your
        # agent containing all the information about the negotiation session.
        if isinstance(data, Settings):
            self.settings = cast(Settings, data)
            self.me = self.settings.getID()

            # progress towards the deadline has to be tracked manually through the use of the Progress object
            self.progress = self.settings.getProgress()

            self.parameters = self.settings.getParameters()
            self.storage_dir = self.parameters.get("storage_dir")

            # the profile contains the preferences of the agent over the domain
            profile_connection = ProfileConnectionFactory.create(
                data.getProfile().getURI(), self.getReporter()
            )
            self.profile = profile_connection.getProfile()
            self.domain = self.profile.getDomain()
            profile_connection.close()

            # set the acceptable bids in a local variable (in second phase of bidding strategy)
            # set the top bids in a local variable (in first phase of bidding strategy)
            all_bids = AllBidsList(self.domain)
            self.acceptable_bids = self.build_best_bids(all_bids, self.acceptable_bids_percent)
            self.best_bids = self.build_best_bids(all_bids, self.best_bids_percent)

        # ActionDone informs you of an action (an offer or an accept)
        # that is performed by one of the agents (including yourself).
        elif isinstance(data, ActionDone):
            action = cast(ActionDone, data).getAction()
            actor = action.getActor()

            # ignore action if it is our action
            if actor != self.me:
                # obtain the name of the opponent, cutting of the position ID.
                self.other = str(actor).rsplit("_", 1)[0]

                # process action done by opponent
                self.opponent_action(action)

        # YourTurn notifies you that it is your turn to act
        elif isinstance(data, YourTurn):
            # execute a turn
            self.my_turn()

        # Finished will be send if the negotiation has ended (through agreement or deadline)
        elif isinstance(data, Finished):
            self.save_data()
            # terminate the agent MUST BE CALLED
            self.logger.log(logging.INFO, "party is terminating:")
            super().terminate()
        else:
            self.logger.log(logging.WARNING, "Ignoring unknown info " + str(data))

    def getCapabilities(self) -> Capabilities:
        """
        Method to indicate to the protocol what the capabilities of this agent are.
        Leave it as is for the ANL 2022 competition

        Returns:
            Capabilities: Capabilities representation class
        """
        return Capabilities(
            set(["SAOP"]),
            set(["geniusweb.profile.utilityspace.LinearAdditive"]),
        )

    def send_action(self, action: Action):
        """Sends an action to the opponent(s)

        Args:
            action (Action): action of this agent
        """
        self.getConnection().send(action)

    # give a description of your agent
    def getDescription(self) -> str:
        """
        Returns a description of your agent. 1 or 2 sentences.

        Returns:
            str: Agent description
        """
        return "Negotiating agent implementation for group 52."

    def opponent_action(self, action):
        """Process an action that was received from the opponent.

        Args:
            action (Action): action of opponent
        """
        # if it is an offer, set the last received bid
        if isinstance(action, Offer):
            # create opponent model if it was not yet initialised
            if self.opponent_model is None:
                self.opponent_model = OpponentModel(self.domain)

            bid = cast(Offer, action).getBid()

            # update opponent model with bid
            self.opponent_model.update(bid)
            # set bid as last received and accumulate all bids in
            # a dedicated list
            self.last_received_bid = bid
            self.opp_bid_tracker.append(bid)

            # update best received bid, according to my own utility
            if (self.best_received_bid is None) or \
                (self.profile.getUtility(bid) > self.profile.getUtility(self.best_received_bid)): 
                self.best_received_bid = bid
                self.logger.log(logging.INFO, "updated best received offer to" + str(self.profile.getUtility(bid)))

    def my_turn(self):
        """
        This method is called when it is our turn. It should decide upon an action
        to perform and send this action to the opponent.
        """
        # we first compute the counterbid the agent wanted to do if the offer is not
        # good enough for us
        self.analyze_negotiation_trends()
        next_bid = self.find_bid()

        # check if the last received offer is good enough
        if self.accept_condition(self.last_received_bid, next_bid):
            # if so, accept the offer
            action = Accept(self.me, self.last_received_bid)
            self.send_action(action)
            return
        
        # if not, send the bid you computed before    
        action = Offer(self.me, next_bid)
        self.send_action(action)

    def save_data(self):
        """
        This method is called after the negotiation is finished. It can be used to store data
        for learning capabilities. Note that no extensive calculations can be done within this method.
        Taking too much time might result in your agent being killed, so use it for storage only.
        """
        data = "Data for learning (see README.md)"
        with open(f"{self.storage_dir}/data.md", "w") as f:
            f.write(data)

    def accept_condition(self, bid: Bid, next_bid: Bid) -> bool:
        if bid is None:
            return False
        
        progress = self.progress.get(time() * 1000)
        
        # PHASE 1: EXPLORATION
        # if we get a very good counteroffer, we accept it; otherwise never
        if(progress < self.exploration_thresh):
            return self.profile.getUtility(bid) >= self.exploration_accept_value
        
        # PHASE 2: "REAL" NEGOTIATION
        # also includes part of phase 3. Here we implement the accept next accepting strategy
        if(progress >= self.exploration_thresh and progress < self.very_last_moments_thresh):
            return self.profile.getUtility(bid) >= self.profile.getUtility(next_bid)
        
        # PHASE 3 (LAST PART): ACCEPT ALL
        # if we have reached the very end without anything, I accept any incoming offer if it's
        # better than having no deal
        if(progress >= self.very_last_moments_thresh):
            res_bid = self.profile.getReservationBid()
            if res_bid is not None:
                if self.profile.getUtility(bid) < self.profile.getUtility(res_bid):
                    return False
            return True

    def find_bid(self) -> Bid:
        progress = self.progress.get(time() * 1000)

        # CHANGE self.score and self.enhanced_score to switch between methods

        # PHASE 1: EXPLORATION
        # if we are in the exploration phase, we pick one randomly chosen bid from 
        # our top bids
        if(progress < self.exploration_thresh):
            self.logger.log(logging.INFO, "exploring...")
            random_bid = randint(0, len(self.best_bids)-1)
            return self.best_bids[random_bid][0]
        
        # PHASE 2: "REAL" NEGOTIATION
        # among the acceptable bids, we select the one which maximizes the score function
        if(progress >= self.exploration_thresh and progress < self.last_moments_thresh):
            self.logger.log(logging.INFO, "negotiating...")
            self.update_alpha()
            bid = max(self.acceptable_bids, key=lambda bid: self.score(bid[0]))
            score = self.score(bid[0])
            best_score_bids = list(filter(lambda b: score - self.tolerance <= self.score(b[0]) <= score, self.acceptable_bids))
            random_bid = randint(0, len(best_score_bids)-1)
            return best_score_bids[random_bid][0]
        
        # PHASE 3: LAST MOMENT BIDS
        # if I reach the end of the time available without an agreement, I propose to the opponent
        # the best bid he has made in the past, if it is better than the reservation value. If not,
        # return the same thing of phase 2
        if(progress >= self.last_moments_thresh):
            self.logger.log(logging.INFO, "last moments...")
            res_bid = self.profile.getReservationBid()
            if res_bid is not None:
                if self.profile.getUtility(self.best_received_bid) <= self.profile.getUtility(res_bid):
                    self.update_alpha()
                    bid = max(self.acceptable_bids, key=lambda bid: self.score(bid[0]))
                    return bid[0]
            return self.best_received_bid
    
    def score(self, bid: Bid) -> float:
        """
        Calculate heuristic score for a bid. This is done by the weighted average of the agent's 
        and the opponent's utilities. The weights of the weighted average change with time.
        """
        our_utility = float(self.profile.getUtility(bid))
        opponent_utility = self.opponent_model.get_predicted_utility(bid)

        return self.alpha * our_utility + (1-self.alpha) * opponent_utility
    
    def update_alpha(self):
        """
        Updates the alpha (for the score computation) with time and according to how the opponent
        behaves. In this way we change how much we take in consideration the opponent's behaviour,
        flexibly depending on how much the opponent has taken our interests in consideration
        """
        current_time = self.progress.get(time() * 1000)

        # we have to build a different line whose slope depends on the behaviour of the opponent.
        # A line has to pass thru the (time_old, alpha) point, and then, with a given slope, it
        # should compute the new alpha' in given x = time (the current time).
        # Here we have the logic for computing m (slope)
        m = 0.75 # default value
        # we first classify the opponent behaviour in recent and less recent times as bad or
        # good for us
        if self.macro_opp_move_type is not None and self.micro_opp_move_type is not None:
        # if they're both none, I stick to the default value
            if self.macro_opp_move_type in ["fortunate", "concession", "nice"]:
                macro = "good"
            else:
                macro = "bad"
            if self.micro_opp_move_type in ["fortunate", "concession", "nice"]:
                micro = "good"
            else:
                micro = "bad"        
            self.logger.log(logging.INFO, f"the opponent has been acting {macro} in the last rounds")
            self.logger.log(logging.INFO, f"the opponent has been acting {micro} in the most recent rounds")

            if (micro, macro) == ("good", "good"):
                # this means the opponent has been behaving well consistently in many of the last rounds of
                # the negotiation; then we want to be concessive too and decrease alpha much so to 
                # take his preferences into consideration
                m = 1.5
            elif (micro, macro) == ("good", "bad"):
                # this means the opponent has been acting well consistently (remember we are working
                # on trends not single moves) in the most recent rounds. We want to be somewhat concessive
                # but not too much, also to trigger an intelligent response
                m = 1
            elif (micro, macro) == ("bad", "bad"):
                # the opponent has been acting bad for a while and doesn't seem to help us in any way.
                # we should treat him the same way
                m = 0.25
            # last case ("bad", "good"): the opponent did act well in the past, but now he's starting to act 
            # bad. We use the default value of m here

        # we compute the new alpha using a linear function. The line has the slope found before
        # and it passes thru the previous (time, alpha) pair (computed in previous iterations).
        # So the formula is y - alpha_old = m * (current_time - old_time).
        # It shouldn't go below a certain threshold tho
        self.alpha = max(self.alpha_min, self.alpha - m * (current_time - self.time_old))

        self.logger.log(logging.INFO, "alpha value updated to" + str(self.alpha))
        self.time_old = current_time
    
    def build_best_bids(self, all_bids, percentage) -> list[tuple[Bid, float]]:
        """
            Computes the top bids available to the agent in the current domain and considering
            its preferences. Among the top bids, we never include bids which are below the reservation
            value utility (we never want them).

            The logic to compute bids is (for now) only positional, meaning on all bids we take the
            first ... %.

            We could have used BidsWithUtils class, but it's deprecated in the docs.
        """
        bids_utils = []

        for i in range(all_bids.size()):
            bid = all_bids.get(i)
            utility = float(self.profile.getUtility(bid))
            bids_utils.append((bid, utility))
        
        # we exclude any bid which is lower than the reservation value (it may happen that
        # the reservation value is high and we end up including bad bids). This if there is 
        # a reservation value
        if self.profile.getReservationBid() is not None:
            bids_utils = list(filter(lambda bid: 
                                    bid[1] > self.profile.getUtility(self.profile.getReservationBid()), 
                                    bids_utils)
                                    )
        
        bids_utils.sort(key=lambda bid: bid[1], reverse=True)

        number_top = ceil(len(bids_utils) * percentage)

        self.logger.log(logging.INFO, "consider only the top" + str(number_top) + "bids")
        self.logger.log(logging.INFO, "upper bound for utility is" + str(bids_utils[0][1]))
        self.logger.log(logging.INFO, "lower bound for utility is" + str(bids_utils[number_top][1]))

        return bids_utils[:number_top]

    def analyze_negotiation_trends(self):
            """
            While the opponent model aims at representing the opponent's preferences and utility
            function, this function here aims at analyzing the actual behaviour of the opponent during
            the negotiation. To do so, we consider the most recent opponent bids and we see how on average
            the behaviour has shifted from the first half to the second half of the bids: this by considering
            the average utilities of both agents in the first part and comparing it with the average utilities
            of the second part. These are compared in the same way step-wise analysis does, and
            depending on how they change we can recognize certain "trends" in the opponent.
            Working on the "bigger picture" instead of on individual moves allows us to characterize the 
            opponent behaviour better, since we have better evidence of bahavioural change.
            """

            # the analysis is done on two levels of "recency"
            analysis_types = ['macro', 'micro']

            # we can operate on a window of the minimum size only if we have enough collected 
            # opponent bids
            if len(self.opp_bid_tracker) < getattr(self, f"{analysis_types[1]}_win"):
                return

            # Extract utilities from both agents, for each bid in bid_tracker
            opponent_utilities = [self.opponent_model.get_predicted_utility(bid) for bid in self.opp_bid_tracker]
            our_utilities = [self.profile.getUtility(bid) for bid in self.opp_bid_tracker]

            for analysis_type in analysis_types:
                window_size = getattr(self, f"{analysis_type}_win")

                # this is in case we don't have enough bids for the macro window, but we do
                # for the micro
                if len(self.opp_bid_tracker) < window_size:
                    continue

                opponent_avg_1 = []  # Opponent's previous utilities
                opponent_avg_2 = []  # Opponent's most recent utilities
                our_avg_1 = []       # Our previous utilities
                our_avg_2 = []       # Our most recent utilities

                # Calculate starting index for this window size
                start_index = max(len(opponent_utilities) - window_size, 0)

                # Loop through the bids in the current window; the first half are going
                # to processed separately from the second half
                for i in range(start_index, len(opponent_utilities)):
                    if i >= start_index + int(window_size / 2):
                        opponent_avg_2.append(opponent_utilities[i])
                        our_avg_2.append(our_utilities[i])
                    else:
                        opponent_avg_1.append(opponent_utilities[i])
                        our_avg_1.append(our_utilities[i])

                # Calculate averages for both halves of the window
                opponent_first_avg = self.average(opponent_avg_1) 
                opponent_second_avg = self.average(opponent_avg_2) 
                our_first_avg = self.average(our_avg_1) 
                our_second_avg = self.average(our_avg_2) 

                # Calculate differences
                opponent_change = opponent_second_avg - opponent_first_avg
                our_change = our_second_avg - our_first_avg

                # Determine move type (actually a trend, but analyze using the
                # DANS framework step-wise classification)
                if opponent_change > 0 and our_change <= 0:
                    move_type = 'selfish'
                elif opponent_change < 0 and our_change >= 0:
                    move_type = 'concession'
                elif opponent_change > 0 and our_change > 0:
                    move_type = 'fortunate'
                elif opponent_change <= 0 and our_change < 0:
                    move_type = 'unfortunate'
                elif opponent_change == 0 and our_change > 0:
                    move_type = 'nice'
                elif opponent_change == 0 and our_change == 0:
                    move_type = 'silent'
                else:
                    move_type = 'neutral'

                # Set the move type as an attribute for both macro and micro analysis
                setattr(self, f"{analysis_type}_opp_move_type", move_type)
    
    def average(self, arr):
        return sum(arr)/len(arr) if arr else 0
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
from geniusweb.bidspace.BidsWithUtility import BidsWithUtility
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
    Template of a Python geniusweb agent.
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
        self.last_moments_thresh = 0.85
        self.very_last_moments_thresh = 0.9

        # this represents the level from which we consider bids in the first phase; we take only
        # ... % best, according to this parameter
        self.best_bids_percent = 1/100 # TO DO implement a better logic
        self.best_bids = None

        # similar as above, but we consider a lower threshold in the second phase
        self.acceptable_bids_percent = 1/5 # TO DO implement a better logic
        self.acceptable_bids = None

        # if we receive a bid with this utility value for us or above in the exploration phase,
        # we accept it
        self.exploration_accept_value = 0.95

        # stores the best opponent's bid from our utility pov
        self.best_received_bid = None

        # these are the initial and final values for the weighted sum of utilities we have to do in the
        # "real" negotiation phase
        self.alpha = 0.8
        self.alpha_max = self.alpha
        self.alpha_min = 0.5

        #opponent tracker
        self.opp_macro_change = []
        self.opp_micro_change = []
        self.macro_win = 100
        self.macro_opp_avg = 0
        self.macro_opp_std = 0
        self.macro_diff_avg = 0
        self.macro_diff_std = 0
        self.micro_opp_std = 0
        self.micro_opp_trend= 0 
        self.micro_diff_std = 0
        self.micro_diff_trend = 0
        self.opp_bid_tracker = []

    def notifyChange(self, data: Inform):
        """
        This is the entry point of all interaction with your agent after is has been initialised.
        How to handle the received data is based on its class type.

        Args:
            info (Inform): Contains either a request for action or information.
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

            # only for debugging purposes
            if self.profile.getReservationBid() is not None:
                print('ao porco dio ci sta il reservation value!')
                self.logger.log(logging.INFO, "reservation value is " + 
                                str(self.profile.getUtility(self.profile.getReservationBid())))

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
            # only for debugging purposes
            if self.profile.getReservationBid() is not None:
                print('ao porco dio ci sta il reservation value!')
                self.logger.log(logging.INFO, "reservation value is " + 
                                str(self.profile.getUtility(self.profile.getReservationBid())))

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
            # set bid as last received
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
            bid = max(self.acceptable_bids, key=lambda bid: self.enhanced_score(bid[0]))
            return bid[0]
        
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
                    bid = max(self.acceptable_bids, key=lambda bid: self.enhanced_score(bid[0]))
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
        Updates the alpha (for the score computation) with time. For now, let's just use a quadratic
        function (but it may change: experiment with this!)
        """
        x = self.progress.get(time() * 1000)

        # we start from 0.8 and we want to get at most to 0.5. We use a quadratic function so that
        # when we are at 80% of the negotiation we have reached the 0.5 value for alpha.
        # self.alpha = max(self.alpha_min, -3.25*(x**2) + 3.225*x)
        self.alpha = max(self.alpha_min, self.alpha_max - x + self.exploration_thresh) # can try with linear
        # self.alpha = max(0.5, self.alpha_max - ln(x+self.exploration_thresh)) # logarithmic

        self.logger.log(logging.INFO, "alpha value updated to" + str(self.alpha))
    
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
        if len(self.opp_bid_tracker) < self.macro_win:
            return

        # Extract utilities
        opponent_utilities = [self.opponent_model.getUtility(bid) for bid in self.opp_bid_tracker]
        our_utilities = [self.our_model.getUtility(bid) for bid in self.opp_bid_tracker]

        # Function to calculate standard deviation
        def calculate_standard_deviation(values):
            """Calculate the standard deviation of utility values."""
            if len(values) < 2:  # Standard deviation is not defined for less than 2 values
                return 0
            mean = sum(values) / len(values)
            variance = sum((x - mean) ** 2 for x in values) / (len(values) - 1)
            return variance ** 0.5

        # Function to calculate slope (trend) of values
        def calculate_slope(values):
            if len(values) < 2:
                return 0
            mean_x = sum(range(len(values))) / len(values)
            mean_y = sum(values) / len(values)
            numerator = sum((x - mean_x) * (y - mean_y) for x, y in enumerate(values))
            denominator = sum((x - mean_x) ** 2 for x in range(len(values)))
            return numerator / denominator if denominator != 0 else 0

        # Macro analysis
        macro_opp_avg = sum(opponent_utilities) / len(opponent_utilities)
        macro_opp_std = calculate_standard_deviation(opponent_utilities)

        macro_diffs = [opponent_utilities[i] - our_utilities[i] for i in range(len(opponent_utilities))]
        macro_diff_avg = sum(macro_diffs) / len(macro_diffs)
        macro_diff_std = calculate_standard_deviation(macro_diffs)

        # Micro analysis (last part of the window)
        last_macro_opp_utilities = opponent_utilities[-self.macro_win:]
        micro_opp_std = calculate_standard_deviation(last_macro_opp_utilities)
        micro_opp_trend = calculate_slope(last_macro_opp_utilities)

        last_macro_diffs = macro_diffs[-self.macro_win:]
        micro_diff_std = calculate_standard_deviation(last_macro_diffs)
        micro_diff_trend = calculate_slope(last_macro_diffs)

        # Update tracking with calculated values
        self.macro_opp_avg, self.macro_opp_std = macro_opp_avg, macro_opp_std
        self.macro_diff_avg, self.macro_diff_std = macro_diff_avg, macro_diff_std
        self.micro_opp_std, self.micro_opp_trend = micro_opp_std, micro_opp_trend
        self.micro_diff_std, self.micro_diff_trend = micro_diff_std, micro_diff_trend

    def enhanced_score(self, bid):
        """
        Calculate an enhanced heuristic score for a bid, incorporating both macro and micro analysis insights.
        This is achieved by a dynamic weighted average of the agent's and the opponent's utilities, 
        where weights are adjusted over time based on negotiation trends.

        :param bid: The bid to evaluate.
        :return: The calculated score for the given bid.
        """
        # Basic utilities
        our_utility = float(self.profile.getUtility(bid))
        opponent_utility = self.opponent_model.get_predicted_utility(bid)
        
        # Macro adjustments
        # Alpha is adjusted based on macro-level insights (e.g., trends in utility standard deviations)
        # A possible adjustment: if our standard deviation is low but the opponent's is high, we might prioritize our utility less
        macro_adjustment = (self.macro_opp_std - self.macro_diff_std) / (self.macro_opp_std + self.macro_diff_std + 0.01)  # Avoid division by zero
        alpha = self.alpha + macro_adjustment * (1 - self.alpha)  # Adjust alpha based on macro insights

        # Micro adjustments
        # Beta reflects adjustments based on micro-level trends, such as recent changes in negotiation dynamics
        # A possible strategy: if the trend in differences is positive, we might give more weight to opponent utility, anticipating their increasing satisfaction
        # Also the difference and opp utility in micro sense indicates how our opponent responds to our bids
        micro_trend_adjustment = self.micro_diff_trend  # This could be scaled or transformed as needed
        beta = (1 - alpha) + micro_trend_adjustment * alpha  # Adjust beta based on micro insights

        # Ensuring alpha and beta remain valid probabilities
        alpha = min(max(alpha, 0), 1)
        beta = min(max(beta, 0), 1 - alpha)

        # Calculate the enhanced score
        return alpha * our_utility + beta * opponent_utility

import json
import time
from pathlib import Path

from utils.plot_trace import plot_trace
from utils.runners import run_session


def run(domain, profile):
    print("")
    print(f"working on domain {domain} with preference profile {profile}")
    print("")

    RESULTS_DIR = Path("results", f"itself_{domain}_{profile}")

    # create results directory if it does not exist
    if not RESULTS_DIR.exists():
        RESULTS_DIR.mkdir(parents=True)

    # Settings to run a negotiation session:
    settings = {
        "agents": [
            {
                "class": "agents.group52_agent.group52_agent.Group52Agent",
                "parameters": {"storage_dir": "agent_storage/Group52Agent"},
            },
            {
                "class": "agents.group52bis_agent.group52bis_agent.Group52BisAgent",
                "parameters": {"storage_dir": "agent_storage/Group52BisAgent"},
            },
        ],
        "profiles": [f"domains/domain{domain}/profileA.json", f"domains/domain{domain}/profileB.json"] if profile=='A'
         else [f"domains/domain{domain}/profileB.json", f"domains/domain{domain}/profileA.json"],
        "deadline_time_ms": 10000,
    }

    # run a session and obtain results in dictionaries
    session_results_trace, session_results_summary = run_session(settings)

    # plot trace to html file
    if not session_results_trace["error"]:
        plot_trace(session_results_trace, RESULTS_DIR.joinpath("trace_plot.html"))

    # write results to file
    with open(RESULTS_DIR.joinpath("session_results_trace.json"), "w", encoding="utf-8") as f:
        f.write(json.dumps(session_results_trace, indent=2))
    with open(RESULTS_DIR.joinpath("session_results_summary.json"), "w", encoding="utf-8") as f:
        f.write(json.dumps(session_results_summary, indent=2))


if __name__ == '__main__':
    domains = ['00', '01', '02', '03', '08', '12', '15', '16', '18', '19', '21',
               '23', '25', '27', '29', '30', '34', '37', '44', '45']

    for domain in domains:
        run(domain, 'A')
        run(domain, 'B')
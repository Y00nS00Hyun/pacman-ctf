def get_performance(times):
    return sum(times) / len(times), max(times)

if __name__ == '__main__':
    import capture
    import argparse

    parser = argparse.ArgumentParser(description="Test Pacman Team")
    parser.add_argument("-q", action="store_true", help="Quiet mode", default=False)
    args = parser.parse_args()

    quiet_mode = args.q
    user_team_name = 'myTeam'
    baseline_team = 'baselineTeam'
    
    games_per_side = 10
    total_games = games_per_side * 2
    win_threshold = 13
    win_count = 0
    game_summaries = []
    print(f"### Testing Team: {user_team_name} ###")
    print(f"### Pass condition: win rate >= {win_threshold/total_games*100:.1f}% ###\n")

    print("--- Playing as RED team ---")
    red_args = [
    '--red', user_team_name, 
    '--blue', baseline_team,
    '-a',
    '--record'
    ]
    red_args.append('--record')
    if quiet_mode:
        red_args.append('-q')

    for i in range(games_per_side):
        print("\n" + "="*30)
        options = capture.readCommand(red_args)
        options['redTeamName'] = user_team_name
        options['blueTeamName'] = baseline_team
        games, timeList = capture.runGames(**options)
        if timeList == None:
            print("Abnormal termination!!")
            continue
        avg0, max0 = get_performance(timeList[0])
        avg2, max2 = get_performance(timeList[2])
        is_win = games[0].state.data.score > 0
        if is_win: win_count += 1
        
        game_summaries.append({
            'game_no': i + 1,
            'role': 'RED',
            'result': 'WIN' if is_win else 'LOSS',
            'score': games[0].state.data.score,
            'stats': [(avg0 + avg2) / 2, max(max0, max2)]
        })
        
        print(f"Progress: {len(game_summaries)}/{total_games} games finished. You won {win_count} times.", end='\r')

    print("\n--- Playing as BLUE team ---")
    blue_args = [
    '--red', baseline_team, 
    '--blue', user_team_name,
    '-a',
    '--record'
    ]
    if quiet_mode:
        blue_args.append('-q')

    for i in range(games_per_side):
        print("\n" + "="*30)
        options = capture.readCommand(blue_args)
        options['redTeamName'] = baseline_team
        options['blueTeamName'] = user_team_name
        games, timeList = capture.runGames(**options)
        if timeList == None:
            print("Abnormal termination!!")
            continue
        avg1, max1 = get_performance(timeList[1])
        avg3, max3 = get_performance(timeList[3])
        is_win = games[0].state.data.score < 0
        if is_win: win_count += 1
        
        game_summaries.append({
            'game_no': i + 1 + games_per_side,
            'role': 'BLUE',
            'result': 'WIN' if is_win else 'LOSS',
            'score': games[0].state.data.score,
            'stats': [(avg1 + avg3) / 2, max(max1, max3)]
        })
        print(f"Progress: {len(game_summaries)}/{total_games} games finished. You won {win_count} times.", end='\r')

    print("\n\n" + "="*76)
    print(f"{'No':<4} | {'Role':<5} | {'Result':<6} | {'Score':<6} | {'Time per step (Avg/Max)':<35}")
    print("-" * 76)

    for res in game_summaries:
        stats = f"{res['stats'][0]:.3f}/{res['stats'][1]:.3f}s"
        print(f"{res['game_no']:<4} | {res['role']:<5} | {res['result']:<6} | {res['score']:<6} | {stats:<35}")

    print("-" * 76)
    win_rate = win_count / total_games
    print(f"FINAL WIN RATE: {win_rate:.2%} ({win_count}/{total_games} wins)")
    print(f"PASS CONDITION: {win_threshold} wins")
    
    if win_count >= win_threshold:
        print(f"Result: PASS! Your team win with BaselineTeam.")
    else:
        print(f"Result: FAIL (Need {win_threshold} or more wins)")
    print("="*30)
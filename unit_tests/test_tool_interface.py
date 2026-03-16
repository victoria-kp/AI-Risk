"""Unit tests for risk_env/tool_interface.py"""

import sys
import os
import unittest
import random

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'pyrisk_vendor'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from game import Game
from world import CONNECT, MAP, KEY, AREAS
from ai.stupid import StupidAI
from risk_env.tool_interface import (
    parse_tool_call, dispatch_tool, run_tool_loop,
    TOOL_REGISTRY, _parse_kwargs, _parse_value, _format_result,
)


def make_game(n_players=3, seed=42, turns=6):
    """Helper: create a pyrisk game and simulate some turns."""
    random.seed(seed)
    g = Game(curses=False, connect=CONNECT, cmap=MAP, ckey=KEY, areas=AREAS)
    names = ['RED', 'BLUE', 'GREEN', 'YELLOW', 'PURPLE'][:n_players]
    for name in names:
        g.add_player(name, StupidAI)
    g.turn_order = list(g.players)
    for i, name in enumerate(g.turn_order):
        g.players[name].color = i + 1
        g.players[name].ord = ord('x')
        g.players[name].ai.start()
    g.initial_placement()
    for _ in range(turns):
        if g.player.alive:
            choices = g.player.ai.reinforce(g.player.reinforcements)
            for tt, ff in choices.items():
                t = g.world.territory(tt)
                if t and t.owner == g.player:
                    t.forces += int(ff)
            for src, target, attack, move in g.player.ai.attack():
                st = g.world.territory(src)
                tt = g.world.territory(target)
                if (st and tt and st.owner == g.player
                        and tt.owner != g.player and tt in st.connect):
                    g.combat(st, tt, attack, move)
        g.turn += 1
    return g


class TestToolRegistry(unittest.TestCase):
    """Verify the tool registry has the expected tools."""

    def test_has_three_tools(self):
        self.assertEqual(len(TOOL_REGISTRY), 3)

    def test_has_battle_sim(self):
        self.assertIn('battle_sim', TOOL_REGISTRY)

    def test_has_threat_analyzer(self):
        self.assertIn('threat_analyzer', TOOL_REGISTRY)

    def test_has_position_evaluator(self):
        self.assertIn('position_evaluator', TOOL_REGISTRY)

    def test_values_are_callable(self):
        for name, fn in TOOL_REGISTRY.items():
            self.assertTrue(callable(fn), f'{name} is not callable')


class TestParseValue(unittest.TestCase):
    """Tests for _parse_value."""

    def test_integer(self):
        self.assertEqual(_parse_value('42'), 42)
        self.assertIsInstance(_parse_value('42'), int)

    def test_negative_integer(self):
        self.assertEqual(_parse_value('-5'), -5)
        self.assertIsInstance(_parse_value('-5'), int)

    def test_zero(self):
        self.assertEqual(_parse_value('0'), 0)
        self.assertIsInstance(_parse_value('0'), int)

    def test_float(self):
        self.assertEqual(_parse_value('3.14'), 3.14)
        self.assertIsInstance(_parse_value('3.14'), float)

    def test_double_quoted_string(self):
        self.assertEqual(_parse_value('"hello"'), 'hello')
        self.assertIsInstance(_parse_value('"hello"'), str)

    def test_single_quoted_string(self):
        self.assertEqual(_parse_value("'world'"), 'world')

    def test_unquoted_string(self):
        result = _parse_value('India')
        self.assertEqual(result, 'India')
        self.assertIsInstance(result, str)


class TestParseKwargs(unittest.TestCase):
    """Tests for _parse_kwargs."""

    def test_empty_string(self):
        self.assertEqual(_parse_kwargs(''), {})

    def test_single_int(self):
        result = _parse_kwargs('attacking=8')
        self.assertEqual(result, {'attacking': 8})

    def test_two_ints(self):
        result = _parse_kwargs('attacking=8, defending=3')
        self.assertEqual(result, {'attacking': 8, 'defending': 3})

    def test_float_value(self):
        result = _parse_kwargs('threshold=0.75')
        self.assertEqual(result, {'threshold': 0.75})

    def test_quoted_string(self):
        result = _parse_kwargs('name="India"')
        self.assertEqual(result, {'name': 'India'})

    def test_single_quoted_string(self):
        result = _parse_kwargs("name='India'")
        self.assertEqual(result, {'name': 'India'})

    def test_mixed_types(self):
        result = _parse_kwargs('attacking=8, name="test", ratio=0.5')
        self.assertEqual(result, {'attacking': 8, 'name': 'test', 'ratio': 0.5})

    def test_extra_spaces(self):
        result = _parse_kwargs('  attacking = 8 ,  defending = 3  ')
        self.assertEqual(result['attacking'], 8)
        self.assertEqual(result['defending'], 3)


class TestParseToolCall(unittest.TestCase):
    """Tests for parse_tool_call."""

    def test_battle_sim_with_args(self):
        text = '<tool_call>battle_sim(attacking=8, defending=3)</tool_call>'
        result = parse_tool_call(text)
        self.assertIsNotNone(result)
        name, kwargs = result
        self.assertEqual(name, 'battle_sim')
        self.assertEqual(kwargs, {'attacking': 8, 'defending': 3})

    def test_threat_analyzer_no_args(self):
        text = '<tool_call>threat_analyzer()</tool_call>'
        result = parse_tool_call(text)
        self.assertIsNotNone(result)
        self.assertEqual(result[0], 'threat_analyzer')
        self.assertEqual(result[1], {})

    def test_position_evaluator_no_args(self):
        text = '<tool_call>position_evaluator()</tool_call>'
        result = parse_tool_call(text)
        self.assertIsNotNone(result)
        self.assertEqual(result[0], 'position_evaluator')

    def test_no_tool_call(self):
        result = parse_tool_call('I think we should attack India.')
        self.assertIsNone(result)

    def test_empty_string(self):
        result = parse_tool_call('')
        self.assertIsNone(result)

    def test_embedded_in_text(self):
        text = ('Let me check the odds. '
                '<tool_call>battle_sim(attacking=10, defending=5)</tool_call> '
                'Then I decide.')
        result = parse_tool_call(text)
        self.assertIsNotNone(result)
        self.assertEqual(result[0], 'battle_sim')
        self.assertEqual(result[1], {'attacking': 10, 'defending': 5})

    def test_at_start(self):
        text = '<tool_call>threat_analyzer()</tool_call> shows threats.'
        result = parse_tool_call(text)
        self.assertEqual(result[0], 'threat_analyzer')

    def test_at_end(self):
        text = 'Checking: <tool_call>position_evaluator()</tool_call>'
        result = parse_tool_call(text)
        self.assertEqual(result[0], 'position_evaluator')

    def test_returns_first_of_multiple(self):
        text = ('<tool_call>battle_sim(attacking=5, defending=2)</tool_call> '
                '<tool_call>threat_analyzer()</tool_call>')
        result = parse_tool_call(text)
        self.assertEqual(result[0], 'battle_sim')

    def test_whitespace_inside_tags(self):
        text = '<tool_call> battle_sim( attacking=8, defending=3 ) </tool_call>'
        result = parse_tool_call(text)
        self.assertIsNotNone(result)
        self.assertEqual(result[0], 'battle_sim')
        self.assertEqual(result[1], {'attacking': 8, 'defending': 3})

    def test_unknown_tool_name(self):
        text = '<tool_call>fake_tool(x=5)</tool_call>'
        result = parse_tool_call(text)
        self.assertIsNotNone(result)
        self.assertEqual(result[0], 'fake_tool')

    def test_multiline_tool_call(self):
        text = '<tool_call>battle_sim(\nattacking=8,\ndefending=3\n)</tool_call>'
        result = parse_tool_call(text)
        self.assertIsNotNone(result)
        self.assertEqual(result[0], 'battle_sim')


class TestFormatResult(unittest.TestCase):
    """Tests for _format_result."""

    def test_simple_dict(self):
        result = _format_result({'win_probability': 0.783, 'remaining': 4.2})
        self.assertIn('win_probability: 0.783', result)
        self.assertIn('remaining: 4.2', result)

    def test_list_of_dicts(self):
        result = _format_result([
            {'territory': 'India', 'score': 3.0},
            {'territory': 'China', 'score': 1.5},
        ])
        self.assertIn('[1]', result)
        self.assertIn('[2]', result)
        self.assertIn('India', result)
        self.assertIn('China', result)

    def test_nested_dict(self):
        result = _format_result({
            'position': {'territories': 14, 'troops': 47},
        })
        self.assertIn('territories: 14', result)
        self.assertIn('troops: 47', result)

    def test_dict_with_list_value(self):
        result = _format_result({
            'targets': [
                {'name': 'Egypt'},
                {'name': 'India'},
            ],
        })
        self.assertIn('Egypt', result)
        self.assertIn('India', result)

    def test_empty_dict(self):
        result = _format_result({})
        self.assertEqual(result, '')

    def test_empty_list(self):
        result = _format_result([])
        self.assertEqual(result, '')

    def test_plain_string(self):
        result = _format_result('hello')
        self.assertEqual(result, 'hello')


class TestDispatchTool(unittest.TestCase):
    """Tests for dispatch_tool."""

    def test_battle_sim(self):
        result = dispatch_tool('battle_sim', {'attacking': 10, 'defending': 3})
        self.assertIn('<tool_result>', result)
        self.assertIn('</tool_result>', result)
        self.assertIn('win_probability', result)

    def test_threat_analyzer(self):
        game = make_game()
        player = game.players['RED']
        result = dispatch_tool('threat_analyzer', {}, game=game, player=player)
        self.assertIn('<tool_result>', result)
        self.assertIn('threat_score', result)

    def test_position_evaluator(self):
        game = make_game()
        player = game.players['RED']
        result = dispatch_tool('position_evaluator', {}, game=game, player=player)
        self.assertIn('<tool_result>', result)
        self.assertIn('territories', result)

    def test_unknown_tool(self):
        result = dispatch_tool('fake_tool', {})
        self.assertIn('<tool_result>', result)
        self.assertIn('Error', result)
        self.assertIn('fake_tool', result)

    def test_unknown_tool_lists_available(self):
        result = dispatch_tool('fake_tool', {})
        self.assertIn('battle_sim', result)
        self.assertIn('threat_analyzer', result)
        self.assertIn('position_evaluator', result)

    def test_bad_args(self):
        result = dispatch_tool('battle_sim', {'wrong_arg': 5})
        self.assertIn('<tool_result>', result)
        self.assertIn('Error', result)

    def test_missing_game_player(self):
        result = dispatch_tool('threat_analyzer', {})
        self.assertIn('<tool_result>', result)
        self.assertIn('Error', result)

    def test_battle_sim_result_has_all_fields(self):
        result = dispatch_tool('battle_sim', {'attacking': 5, 'defending': 3})
        self.assertIn('win_probability', result)
        self.assertIn('expected_attacker_remaining', result)
        self.assertIn('expected_defender_remaining', result)


class TestRunToolLoopSingle(unittest.TestCase):
    """Tests for run_tool_loop with single tool calls."""

    def setUp(self):
        self.game = make_game()
        self.player = self.game.players['RED']

    def test_single_call_replaces(self):
        text = 'Check: <tool_call>battle_sim(attacking=8, defending=3)</tool_call> done.'
        updated, log = run_tool_loop(text, game=self.game, player=self.player)
        self.assertEqual(len(log), 1)
        self.assertNotIn('<tool_call>', updated)
        self.assertIn('<tool_result>', updated)

    def test_preserves_surrounding_text(self):
        text = 'BEFORE <tool_call>battle_sim(attacking=5, defending=2)</tool_call> AFTER'
        updated, log = run_tool_loop(text, game=self.game, player=self.player)
        self.assertIn('BEFORE', updated)
        self.assertIn('AFTER', updated)

    def test_log_has_tool_name(self):
        text = '<tool_call>battle_sim(attacking=5, defending=2)</tool_call>'
        _, log = run_tool_loop(text, game=self.game, player=self.player)
        self.assertEqual(log[0]['tool_name'], 'battle_sim')

    def test_log_has_kwargs(self):
        text = '<tool_call>battle_sim(attacking=5, defending=2)</tool_call>'
        _, log = run_tool_loop(text, game=self.game, player=self.player)
        self.assertEqual(log[0]['kwargs'], {'attacking': 5, 'defending': 2})

    def test_log_has_result(self):
        text = '<tool_call>battle_sim(attacking=5, defending=2)</tool_call>'
        _, log = run_tool_loop(text, game=self.game, player=self.player)
        self.assertIn('<tool_result>', log[0]['result'])

    def test_threat_analyzer_call(self):
        text = '<tool_call>threat_analyzer()</tool_call>'
        updated, log = run_tool_loop(text, game=self.game, player=self.player)
        self.assertEqual(len(log), 1)
        self.assertEqual(log[0]['tool_name'], 'threat_analyzer')
        self.assertIn('threat_score', updated)

    def test_position_evaluator_call(self):
        text = '<tool_call>position_evaluator()</tool_call>'
        updated, log = run_tool_loop(text, game=self.game, player=self.player)
        self.assertEqual(len(log), 1)
        self.assertEqual(log[0]['tool_name'], 'position_evaluator')
        self.assertIn('territories', updated)


class TestRunToolLoopMultiple(unittest.TestCase):
    """Tests for run_tool_loop with multiple tool calls."""

    def setUp(self):
        self.game = make_game()
        self.player = self.game.players['RED']

    def test_two_calls(self):
        text = ('<tool_call>threat_analyzer()</tool_call> '
                '<tool_call>battle_sim(attacking=5, defending=2)</tool_call>')
        updated, log = run_tool_loop(text, game=self.game, player=self.player)
        self.assertEqual(len(log), 2)
        self.assertEqual(log[0]['tool_name'], 'threat_analyzer')
        self.assertEqual(log[1]['tool_name'], 'battle_sim')

    def test_all_calls_replaced(self):
        text = ('<tool_call>threat_analyzer()</tool_call> '
                '<tool_call>battle_sim(attacking=5, defending=2)</tool_call>')
        updated, log = run_tool_loop(text, game=self.game, player=self.player)
        self.assertNotIn('<tool_call>', updated)
        self.assertEqual(updated.count('<tool_result>'), 2)

    def test_three_calls(self):
        text = ('<tool_call>threat_analyzer()</tool_call> '
                '<tool_call>position_evaluator()</tool_call> '
                '<tool_call>battle_sim(attacking=5, defending=2)</tool_call>')
        updated, log = run_tool_loop(text, game=self.game, player=self.player)
        self.assertEqual(len(log), 3)
        self.assertNotIn('<tool_call>', updated)


class TestRunToolLoopMaxCalls(unittest.TestCase):
    """Tests for max_calls parameter."""

    def setUp(self):
        self.game = make_game()
        self.player = self.game.players['RED']
        self.text = ('<tool_call>battle_sim(attacking=3, defending=1)</tool_call> '
                     '<tool_call>battle_sim(attacking=5, defending=2)</tool_call> '
                     '<tool_call>battle_sim(attacking=8, defending=3)</tool_call>')

    def test_max_calls_1(self):
        _, log = run_tool_loop(self.text, game=self.game,
                               player=self.player, max_calls=1)
        self.assertEqual(len(log), 1)

    def test_max_calls_2(self):
        updated, log = run_tool_loop(self.text, game=self.game,
                                     player=self.player, max_calls=2)
        self.assertEqual(len(log), 2)
        self.assertIn('<tool_call>', updated)  # one left unprocessed

    def test_max_calls_3(self):
        updated, log = run_tool_loop(self.text, game=self.game,
                                     player=self.player, max_calls=3)
        self.assertEqual(len(log), 3)
        self.assertNotIn('<tool_call>', updated)


class TestRunToolLoopNoCall(unittest.TestCase):
    """Tests when there are no tool calls."""

    def setUp(self):
        self.game = make_game()
        self.player = self.game.players['RED']

    def test_no_call_returns_same_text(self):
        text = 'I think we should reinforce India.'
        updated, log = run_tool_loop(text, game=self.game, player=self.player)
        self.assertEqual(updated, text)
        self.assertEqual(len(log), 0)

    def test_empty_string(self):
        updated, log = run_tool_loop('', game=self.game, player=self.player)
        self.assertEqual(updated, '')
        self.assertEqual(len(log), 0)


class TestRunToolLoopErrors(unittest.TestCase):
    """Tests for error handling in run_tool_loop."""

    def setUp(self):
        self.game = make_game()
        self.player = self.game.players['RED']

    def test_unknown_tool(self):
        text = '<tool_call>nonexistent_tool(x=5)</tool_call>'
        updated, log = run_tool_loop(text, game=self.game, player=self.player)
        self.assertEqual(len(log), 1)
        self.assertEqual(log[0]['tool_name'], 'nonexistent_tool')
        self.assertIn('Error', updated)
        self.assertNotIn('<tool_call>', updated)

    def test_bad_args_doesnt_crash(self):
        text = '<tool_call>battle_sim(wrong=5)</tool_call>'
        updated, log = run_tool_loop(text, game=self.game, player=self.player)
        self.assertEqual(len(log), 1)
        self.assertIn('Error', updated)
        self.assertNotIn('<tool_call>', updated)

    def test_error_then_valid(self):
        text = ('<tool_call>fake_tool()</tool_call> '
                '<tool_call>battle_sim(attacking=5, defending=2)</tool_call>')
        updated, log = run_tool_loop(text, game=self.game, player=self.player)
        self.assertEqual(len(log), 2)
        self.assertIn('Error', log[0]['result'])
        self.assertIn('win_probability', log[1]['result'])


class TestRunToolLoopRealistic(unittest.TestCase):
    """Test with realistic model outputs."""

    def setUp(self):
        self.game = make_game()
        self.player = self.game.players['RED']

    def test_reasoning_with_tool(self):
        text = """I need to evaluate whether attacking is a good move.
Let me check the battle odds.
<tool_call>battle_sim(attacking=6, defending=2)</tool_call>
Based on these results, I'll make my recommendation."""
        updated, log = run_tool_loop(text, game=self.game, player=self.player)
        self.assertEqual(len(log), 1)
        self.assertNotIn('<tool_call>', updated)
        self.assertIn('<tool_result>', updated)
        self.assertIn('I need to evaluate', updated)
        self.assertIn('make my recommendation', updated)
        self.assertIn('win_probability', updated)

    def test_multi_step_reasoning(self):
        text = """First, let me check my vulnerabilities.
<tool_call>threat_analyzer()</tool_call>
Now let me check if I should attack the weakest neighbor.
<tool_call>battle_sim(attacking=8, defending=3)</tool_call>
Based on both analyses, here is my recommendation."""
        updated, log = run_tool_loop(text, game=self.game, player=self.player)
        self.assertEqual(len(log), 2)
        self.assertNotIn('<tool_call>', updated)
        self.assertEqual(updated.count('<tool_result>'), 2)
        self.assertIn('here is my recommendation', updated)


if __name__ == '__main__':
    unittest.main()

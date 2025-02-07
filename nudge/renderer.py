from datetime import datetime
from typing import Union

import numpy as np
import torch as th
import pygame
import vidmaker

from nudge.agents.logic_agent import NsfrActorCritic
from nudge.agents.neural_agent import ActorCritic
from nudge.utils import load_model, yellow
from nudge.env import NudgeBaseEnv


from nudge.utils import get_program_nsfr
import copy

SCREENSHOTS_BASE_PATH = "out/screenshots/"
PREDICATE_PROBS_COL_WIDTH = 500 * 2
FACT_PROBS_COL_WIDTH = 1000
CELL_BACKGROUND_DEFAULT = np.array([40, 40, 40])
CELL_BACKGROUND_HIGHLIGHT = np.array([40, 150, 255])
CELL_BACKGROUND_HIGHLIGHT_POLICY = np.array([234, 145, 152])
CELL_BACKGROUND_SELECTED = np.array([80, 80, 80])


class Renderer:
    model: Union[NsfrActorCritic, ActorCritic]
    window: pygame.Surface
    clock: pygame.time.Clock

    def __init__(self,
                 agent_path: str = None,
                 env_name: str = "seaquest",
                 device: str = "cpu",
                 fps: int = None,
                 deterministic=True,
                 env_kwargs: dict = None,
                 render_predicate_probs=True,
                 seed=0):

        self.fps = fps
        self.deterministic = deterministic
        self.render_predicate_probs = render_predicate_probs

        # Load model and environment
        self.model = load_model(agent_path, env_kwargs_override=env_kwargs, device=device)
        self.env = NudgeBaseEnv.from_name(env_name, mode='deictic', seed=seed, **env_kwargs)
        # self.env = self.model.env
        self.env.reset()
        
        print(self.model._print())

        print(f"Playing '{self.model.env.name}' with {'' if deterministic else 'non-'}deterministic policy.")

        if fps is None:
            fps = 15
        self.fps = fps

        try:
            self.action_meanings = self.env.env.get_action_meanings()
            self.keys2actions = self.env.env.unwrapped.get_keys_to_action()
        except Exception:
            print(yellow("Info: No key-to-action mapping found for this env. No manual user control possible."))
            self.action_meanings = None
            self.keys2actions = {}
        self.current_keys_down = set()

        self.predicates = self.model.logic_actor.prednames

        self._init_pygame()

        self.running = True
        self.paused = False
        self.fast_forward = False
        self.reset = False
        self.takeover = False
        

    def _init_pygame(self):
        pygame.init()
        pygame.display.set_caption("Environment")
        frame = self.env.env.render()
        self.env_render_shape = frame.shape[:2]
        window_shape = list(self.env_render_shape)
        if self.render_predicate_probs:
            window_shape[0] += PREDICATE_PROBS_COL_WIDTH
        self.window = pygame.display.set_mode(window_shape, pygame.SCALED)
        self.clock = pygame.time.Clock()
        self.font = pygame.font.SysFont('Calibri', 24)


    def run(self):
        length = 0
        ret = 0

        obs, obs_nn = self.env.reset()
        obs_nn = th.tensor(obs_nn, device=self.model.device) 
        # print(obs_nn.shape)

        while self.running:
            
            self.reset = False
            self._handle_user_input()
            if not self.paused:
                if not self.running:
                    break  # outer game loop

                if self.takeover:  # human plays game manually
                    # assert False, "Unimplemented."
                    action = self._get_action()
                else:  # AI plays the game
                    # print("obs_nn: ", obs_nn.shape)
                    action, logprob = self.model.act(obs_nn, obs) # update the model's internals
                    value = self.model.get_value(obs_nn, obs)


                self.action = action # Store the selected action.


                (new_obs, new_obs_nn), reward, done, terminations, infos = self.env.step(action, is_mapped=self.takeover)
                # if reward > 0:
                    # print(f"Reward: {reward:.2f}")
                new_obs_nn = th.tensor(new_obs_nn, device=self.model.device) 


                self.neural_state = new_obs_nn # Store neural state.
                self.logic_state = new_obs

                #print("hi")
                #print(self.model.actor.get_neural_explanation(new_obs_nn, action))
                #print("hiho")

                self._render()


                if self.takeover and float(reward) != 0:
                    print(f"Reward {reward:.2f}")

                if self.reset:
                    done = True
                    new_obs = self.env.reset()
                    self._render()

                obs = new_obs
                obs_nn = new_obs_nn
                length += 1

                if done:
                    print(f"Return: {ret} - Length {length}")
                    ret = 0
                    length = 0
                    self.env.reset()

        pygame.quit()

    def _get_action(self):
        if self.keys2actions is None:
            return 0  # NOOP
        pressed_keys = list(self.current_keys_down)
        pressed_keys.sort()
        pressed_keys = tuple(pressed_keys)
        if pressed_keys in self.keys2actions.keys():
            return self.keys2actions[pressed_keys]
        else:
            return 0  # NOOP

    def _handle_user_input(self):
        events = pygame.event.get()
        for event in events:
            if event.type == pygame.QUIT:  # window close button clicked
                self.running = False

            elif event.type == pygame.KEYDOWN:  # keyboard key pressed
                if event.key == pygame.K_p:  # 'P': pause/resume
                    self.paused = not self.paused

                elif event.key == pygame.K_r:  # 'R': reset
                    self.reset = True

                elif event.key == pygame.K_f:  # 'F': fast forward
                    self.fast_forward = not(self.fast_forward)

                elif event.key == pygame.K_t:  # 'T': trigger takeover
                    if self.takeover:
                        print("AI takeover")
                    else:
                        print("Human takeover")
                    self.takeover = not self.takeover
                
                elif event.key == pygame.K_o:  # 'O': toggle overlay
                    self.env.env.render_oc_overlay = not(self.env.env.render_oc_overlay)

                elif event.key == pygame.K_c:  # 'C': capture screenshot
                    file_name = f"{datetime.strftime(datetime.now(), '%Y-%m-%d-%H-%M-%S')}.png"
                    pygame.image.save(self.window, SCREENSHOTS_BASE_PATH + file_name)

                elif (event.key,) in self.keys2actions.keys():  # env action
                    self.current_keys_down.add(event.key)

            elif event.type == pygame.KEYUP:  # keyboard key released
                if (event.key,) in self.keys2actions.keys():
                    self.current_keys_down.remove(event.key)

                # elif event.key == pygame.K_f:  # 'F': fast forward
                #     self.fast_forward = False

    def _render(self):
        self.window.fill((20, 20, 20))  # clear the entire window
        #self._render_policy_probs()
        #self._render_predicate_probs()
        #self._render_neural_probs()

        #self._render_selected_action() # Render all actions and highlight the raw selected action.
        #self._render_semantic_action() # Render the list of semantic actions and highlight the actions that make up the selected action.
        self._render_logic_rules() # Render the logic action rules and highlight the selected ones.
        self._render_env()

        pygame.display.flip()
        pygame.event.pump()
        if not self.fast_forward:
            self.clock.tick(self.fps)

    def _render_env(self):
        frame = self.env.env.render()
        frame_surface = pygame.Surface(self.env_render_shape)
        pygame.pixelcopy.array_to_surface(frame_surface, frame)
        self.window.blit(frame_surface, (0, 0))

    def _render_policy_probs_rows(self):
        anchor = (self.env_render_shape[0] + 10, 25)

        model = self.model
        policy_names = ['neural', 'logic']
        weights = model.get_policy_weights()
        for i, w_i in enumerate(weights):
            w_i = w_i.item()
            name = policy_names[i]
            # Render cell background
            color = w_i * CELL_BACKGROUND_HIGHLIGHT_POLICY + (1 - w_i) * CELL_BACKGROUND_DEFAULT
            pygame.draw.rect(self.window, color, [
                anchor[0] - 2,
                anchor[1] - 2 + i * 35,
                PREDICATE_PROBS_COL_WIDTH - 12,
                28
            ])
            # print(w_i, name)

            text = self.font.render(str(f"{w_i:.3f} - {name}"), True, "white", None)
            text_rect = text.get_rect()
            text_rect.topleft = (self.env_render_shape[0] + 10, 25 + i * 35)
            self.window.blit(text, text_rect)
            
    def _render_policy_probs(self):
        anchor = (self.env_render_shape[0] + 10, 25)

        model = self.model
        policy_names = ['neural', 'logic']
        weights = model.get_policy_weights()
        for i, w_i in enumerate(weights):
            w_i = w_i.item()
            name = policy_names[i]
            # Render cell background
            color = w_i * CELL_BACKGROUND_HIGHLIGHT_POLICY + (1 - w_i) * CELL_BACKGROUND_DEFAULT
            pygame.draw.rect(self.window, color, [
                anchor[0] - 2 + i * 500,
                anchor[1] - 2,
                (PREDICATE_PROBS_COL_WIDTH / 2 - 12) * w_i,
                28
            ])

            text = self.font.render(str(f"{w_i:.3f} - {name}"), True, "white", None)
            text_rect = text.get_rect()
            if i == 0:
                text_rect.topleft = (self.env_render_shape[0] + 10, 25) 
            else:
                text_rect.topleft = (self.env_render_shape[0] + 10 + i * 500, 25)
            self.window.blit(text, text_rect)
        
    def _render_predicate_probs(self):
        anchor = (self.env_render_shape[0] + 10, 25)
        nsfr = self.model.actor.logic_actor
        pred_vals = {pred: nsfr.get_predicate_valuation(pred, initial_valuation=False) for pred in nsfr.prednames}
        for i, (pred, val) in enumerate(pred_vals.items()):
            i += 2
            # Render cell background
            color = val * CELL_BACKGROUND_HIGHLIGHT + (1 - val) * CELL_BACKGROUND_DEFAULT
            pygame.draw.rect(self.window, color, [
                anchor[0] - 2 + PREDICATE_PROBS_COL_WIDTH / 2,
                anchor[1] - 2 + i * 35,
                (PREDICATE_PROBS_COL_WIDTH /2  - 12) * val,
                28
            ])

            text = self.font.render(str(f"{val:.3f} - {pred}"), True, "white", None)
            text_rect = text.get_rect()
            text_rect.topleft = (self.env_render_shape[0] + 10 + PREDICATE_PROBS_COL_WIDTH / 2, 25 + i * 35)
            self.window.blit(text, text_rect)
            
            
    def _render_neural_probs(self):
        anchor = (self.env_render_shape[0] + 10, 25)
        blender_actor = self.model.actor
        action_vals = blender_actor.neural_action_probs[0].detach().cpu().numpy()
        action_names = ["noop", "fire", "up", "right", "left", "down", "upright", "upleft", "downright", "downleft", "upfire", "rightfire", "leftfire", "downfire", "uprightfire", "upleftfire", "downrightfire", "downleftfire"]
        for i, (pred, val) in enumerate(zip(action_names, action_vals)):
            i += 2
            # Render cell background
            color = val * CELL_BACKGROUND_HIGHLIGHT + (1 - val) * CELL_BACKGROUND_DEFAULT
            pygame.draw.rect(self.window, color, [
                anchor[0] - 2,
                anchor[1] - 2 + i * 35,
                (PREDICATE_PROBS_COL_WIDTH / 2  - 12) * val,
                28
            ])

            text = self.font.render(str(f"{val:.3f} - {pred}"), True, "white", None)
            text_rect = text.get_rect()
            text_rect.topleft = (self.env_render_shape[0] + 10, 25 + i * 35)
            self.window.blit(text, text_rect)

            
    def _render_facts(self, th=0.1):
        anchor = (self.env_render_shape[0] + 10, 25)

        # nsfr = self.nsfr_reasoner
        nsfr = self.model.actor.logic_actor
        
        fact_vals = {}
        v_T = nsfr.V_T[0]
        preds_to_skip = ['.', 'true_predicate', 'test_predicate_global', 'test_predicate_object']
        for i, atom in enumerate(nsfr.atoms):
            if v_T[i] > th:
                if atom.pred.name not in preds_to_skip:
                    fact_vals[atom] = v_T[i].item()
                
        for i, (fact, val) in enumerate(fact_vals.items()):
            i += 2
            # Render cell background
            color = val * CELL_BACKGROUND_HIGHLIGHT + (1 - val) * CELL_BACKGROUND_DEFAULT
            pygame.draw.rect(self.window, color, [
                anchor[0] - 2,
                anchor[1] - 2 + i * 35,
                FACT_PROBS_COL_WIDTH - 12,
                28
            ])

            text = self.font.render(str(f"{val:.3f} - {fact}"), True, "white", None)
            text_rect = text.get_rect()
            text_rect.topleft = (self.env_render_shape[0] + 10, 25 + i * 35)
            self.window.blit(text, text_rect)




    def _render_selected_action(self):
        '''
        Render all possible actions and highlight only the raw selected action.
        '''
        #action_text = f"Raw selected action: {self.action_meanings[self.action]}"
        #text = self.font.render(action_text, True, "white", None) # Display the raw selected action.
        #text_rect = text.get_rect()
        #text_rect.topleft = (self.env_render_shape[0] + 10, 25 + 25 * 35)  # Place it at the bottom.
        #self.window.blit(text, text_rect)

        anchor = (self.env_render_shape[0] + 10, 25)

        action_names = ["noop", "fire", "up", "right", "left", "down", "upright", "upleft", "downright", "downleft", "upfire", "rightfire", "leftfire", "downfire", "uprightfire", "upleftfire", "downrightfire", "downleftfire"]

        title = self.font.render("Raw Selected Action", True, "white", None)
        title_rect = title.get_rect()
        title_rect.topleft = (self.env_render_shape[0] + 10, 25)
        self.window.blit(title, title_rect)

        for i, action in enumerate(action_names):
            is_selected = 0
            if action.upper() == self.action_meanings[self.action]:
                is_selected = 1 # Only the selected action will be highlighted.

            color = is_selected * CELL_BACKGROUND_HIGHLIGHT + (1 - is_selected) * CELL_BACKGROUND_DEFAULT
            i += 2
            # Render cell background
            pygame.draw.rect(self.window, color, [
                anchor[0] - 2,
                anchor[1] - 2 + i * 35,
                (PREDICATE_PROBS_COL_WIDTH / 4  - 12) * is_selected,
                28
            ])

            text = self.font.render(action, True, "white", None)
            text_rect = text.get_rect()
            text_rect.topleft = (self.env_render_shape[0] + 10, 25 + i * 35)
            self.window.blit(text, text_rect)


    def _parse_semantic_action(self, action):
        '''
        Return a list of actions that together make up the given action.
        '''

        action_names = ["noop", "fire", "up", "right", "left", "down"]
        selected_actions = []
        for elem in action_names:
            if elem in self.action_meanings[action].lower():
                selected_actions.append(elem)

        return selected_actions

    def _render_semantic_action(self):
        '''
        Render only semantic actions and highlight the actions that make up the current selected action.
        '''
        anchor = (self.env_render_shape[0] + 10, 25)
        semantic_actions = ["noop", "fire", "up", "right", "left", "down"]
        selected_actions = self._parse_semantic_action(self.action)

        title = self.font.render("Semantic Actions", True, "white", None)
        title_rect = title.get_rect()
        title_rect.topleft = (self.env_render_shape[0] + 10, 25)
        self.window.blit(title, title_rect)

        for i, action in enumerate(semantic_actions):
            is_selected = 0
            if action in selected_actions:
                is_selected = 1

            color = is_selected * CELL_BACKGROUND_HIGHLIGHT + (1 - is_selected) * CELL_BACKGROUND_DEFAULT
            i += 2
            # Render cell background
            pygame.draw.rect(self.window, color, [
                anchor[0] - 2,
                anchor[1] - 2 + i * 35,
                (PREDICATE_PROBS_COL_WIDTH / 4  - 12) * is_selected,
                28
            ])

            text = self.font.render(action, True, "white", None)
            text_rect = text.get_rect()
            text_rect.topleft = (self.env_render_shape[0] + 10, 25 + i * 35)
            self.window.blit(text, text_rect)
        
        action_text = f"Action: {self.action_meanings[self.action]}"
        text = self.font.render(action_text, True, "white", None) # Display the raw selected action.
        text_rect = text.get_rect()
        text_rect.topleft = (self.env_render_shape[0] + 10, 25 + 10 * 35)  # Place it at the bottom.
        self.window.blit(text, text_rect)


    def selected_logic_rule(self):
        predicate_indices = []
        action_logic_prob = 0
        pred2prob_dict = {} # Key is index of the predicate and value is the probability that it has assigned to the action.

        action = self.action_meanings[self.action].lower()
        basic_actions = self.model.actor.env.pred2action.keys()
        action_indices = self.model.actor.env.pred2action 
        action_predicates = self.model.actor.env_action_id_to_action_pred_indices # Dictionary of actions and its predicates.

        if action in basic_actions:
            # If selected action is a basic action, then it has predicates that contributed to its probability distribution.
            predicate_indices = action_predicates[action_indices[action]]
            action_logic_prob = self.model.actor.logic_action_probs[0].tolist()[action_indices[action]] # Probability of the selected action given logic probability distribution.
            
            # Partly taken from to_action_distribution() in blender_agent.py.
            indices = th.tensor(action_predicates[action_indices[action]]) # Indices of the predicates of selected action.
            indices = indices.expand(self.model.actor.batch_size, -1)
            indices = indices.to(self.model.actor.device)
            gathered = th.gather(th.logit(self.model.actor.raw_action_probs, eps=0.01), 1, indices)

            predicate_probs = th.softmax(gathered, dim=1).cpu().detach().numpy()[0] # Normalized probabilities of the predicates of selected action that they have assigned to it.
            for j in range(len(indices.tolist()[0])):
                pred2prob_dict[indices.tolist()[0][j]] = predicate_probs[j]
            #print(action_predicates[action_indices[action]])
        
        return predicate_indices, action_logic_prob, pred2prob_dict
    

    def _render_logic_rules(self):
        '''
        Render logic action rules and highlight the selected rule.
        '''
        anchor = (self.env_render_shape[0] + 10, 25)
        logic_action_rules = get_program_nsfr(self.model.logic_actor)

        title = self.font.render("Logic Action Rules", True, "white", None)
        title_rect = title.get_rect()
        title_rect.topleft = (self.env_render_shape[0] + 10, 25)
        self.window.blit(title, title_rect)

        predicate_indices, action_logic_prob, pred2prob_dict = self.selected_logic_rule()
        
        logic_policy_weight = self.model.actor.w_policy[1].tolist() # Determines how much influence the logic action probabilities have on the overall action probability distribution.

        for i, rule in enumerate(logic_action_rules):
            is_selected = 0
            if i in predicate_indices and self.model.actor.blender_mode != "neural" and action_logic_prob * logic_policy_weight > 0.1 and pred2prob_dict[i] > 0.1:
                # Highlight predicates that participated in the selection of the action with their assignment of a probability bigger than 0.1.
                # Another condition is a large enough weight for the logic policy during its blending with the neural module, as well as logic probability of the action.
                is_selected = 1

            color = is_selected * CELL_BACKGROUND_HIGHLIGHT + (1 - is_selected) * CELL_BACKGROUND_DEFAULT
            i += 2
            # Render cell background
            pygame.draw.rect(self.window, color, [
                anchor[0] - 2,
                anchor[1] - 2 + i * 35,
                (PREDICATE_PROBS_COL_WIDTH / 1.25  - 12) * is_selected,
                28
            ])

            text = self.font.render(rule, True, "white", None)
            text_rect = text.get_rect()
            text_rect.topleft = (self.env_render_shape[0] + 10, 25 + i * 35)
            self.window.blit(text, text_rect)

    
    def all_object_pairs(self, all_rule_state_atoms, grounded_rule, current_obj, predicates_obj, predicates):
        '''
        Recursive function for determining all grounded rules.
        '''

        if all_rule_state_atoms == []: # Base case. All atoms of the grounded rule have been iterated. Append the grounded rule [(right_of_diver(obj1,obj2), 0.7), (visible_diver(obj2), 0.9), (not_full_divers(img), 1)] for example to the list of all rules.
           self.grounded_rule_atoms.append(grounded_rule)
        
        else:
            fixed_obj = 0

            new_atom_obj = predicates_obj[predicates.index(all_rule_state_atoms[0][0][0].pred.name)].terms # List of objects of the new atom. If it is right_of_diver, then the list is [P, D].

            obj_indices = [] # Indices of objects that are already set by previous atoms. If a previous atom has set {P: obj1} in current_obj, then this list contains [0] if P is the first key in the dictionary.

            for obj in new_atom_obj:
                if obj in current_obj:
                    fixed_obj += 1
                    obj_indices.append(list(current_obj.keys()).index(obj))
            
            if fixed_obj == len(new_atom_obj): # All objects of the state atom are already set by previous atoms and have to be the same. There can only be one grounded atom that fulfills this condition in the list. For example, atom is right_of_diver(P, D) and the objects are already in current_obj.
                for atom in all_rule_state_atoms[0]: # Iterate over atom list. [(right_of_diver(obj1,obj2), 0.7), (right_of_diver(obj1,obj3), 0.8), (right_of_diver(obj1,obj4), 0.2), (right_of_diver(obj1,obj5), 0.1)]
                    if all(atom_obj in current_obj.values() for atom_obj in atom[0].terms): # If all objects of atom match values of the current_obj dictionary, then atom is appended to grounded rule.
                        grounded_rule.append(atom)
                        self.all_object_pairs(all_rule_state_atoms[1:], grounded_rule, current_obj, predicates_obj, predicates) # Recursion, go over next atoms with list that contains newly added atom. E.g. all_rule_state_atoms now is [[(visible_diver(obj2), 0.9), (visible_diver(obj3), 0.3) (visible_diver(obj4), 0.1)], [(not_full_divers(img), 1)]]
                    
            elif fixed_obj < len(new_atom_obj): # At least one object of the state atom has not been set by previous atoms. Multiple grounded atoms can fulfill this condition, therefore every combination has to be checked. E.g. P and D are set, but atom has object X.
                for atom in all_rule_state_atoms[0]:
                    # Go through all grounded atoms of the current atom and see which object combinations match.
                    matching_obj = 0
                    new_obj_indices = []
                    for i, atom_obj in enumerate(atom[0].terms):
                        # Check if atom has correct number of matching objects to current_obj as it's supposed to. E.g. if P and D have been set, and we have an atom that has objects P and X, then one should match.
                        if atom_obj in current_obj.values():
                            matching_obj += 1
                        else:
                            new_obj_indices.append(i)
                    if matching_obj == fixed_obj:
                        new_current_obj = copy.deepcopy(current_obj)
                        new_grounded_rule = copy.deepcopy(grounded_rule)
                        for index in new_obj_indices:
                            new_current_obj[new_atom_obj[index]] = atom[0].terms[index] # Add new objects to current_obj dictionary. {P: obj1, D: obj3, X: obj5}

                        new_grounded_rule.append(atom) # Update grounded rule list with new atom. E.g. [(right_of_diver(obj1,obj2), 0.7), (visible_diver(obj2), 0.9)]
                        
                        self.all_object_pairs(all_rule_state_atoms[1:], new_grounded_rule, new_current_obj, predicates_obj, predicates)


    def _render_logic_valuations(self):
        '''
        Render logic state valuations by highlighting the top 3 logic action rules and showing the truth values of their state predicates.
        '''
        anchor = (self.env_render_shape[0] + 10, 25)

        #title = self.font.render("Logic State Valuations", True, "white", None)
        #title_rect = title.get_rect()
        #title_rect.topleft = (self.env_render_shape[0] + 10, 25)
        #self.window.blit(title, title_rect)

        # Get the current logic valuations
        valuation = self.model.logic_actor.V_T
        batch = valuation[0].detach().cpu().numpy()  # List of values of all state atoms.

        rules_and_predicates = {} # Contains logic action rules and their corresponding state atoms. {up_ladder(X): [on_ladder(P,L), same_level_ladder(P,L)], ...}

        grounded_rules = {} # Contains each logic action rule and their highest grounded rule. E.g. {up_ladder: [(on_ladder(x1,x2), 0.8), (same_level_ladder(x1,x2), 0.9)], left_ladder: [(right_of_ladder(x1,x2), 0.2), (same_level_ladder(x1,x2), 0.9)], right_ladder: [(left_of_ladder(x1,x2), 0.9), (same_level_ladder(x1,x2), 0.9)]}

        for clause in self.model.logic_actor.clauses:
            # Go through every logic action rule.
            predicates_obj = []
            predicates = []
            for predicate in clause.body:
                # Collect the state atoms of the logic action rule in a list.
                predicates_obj.append(predicate)
                predicates.append(predicate.pred.name)
            
            rules_and_predicates[clause.head] = predicates_obj

            objects = set() # Set of objects of the clause. {P,L}
            for predicate in rules_and_predicates[clause.head]:
                objects.update(predicate.terms)


            all_rule_state_atoms = [] # Contains all state atoms of the rule with all object combinations. List of lists. [[(on_ladder(x1,x2), 0.8), (on_ladder(x1,x3), 0.4)], [(same_level_ladder(x1,x2), 0.9), (same_level_ladder(x1,x3), 0.2)]] for rule up_ladder.

            for predicate in predicates_obj:
                rule_atom = [] # List of a state atom of the current rule for all object combinations. [(on_ladder(x1,x2), 0.8), (on_ladder(x1,x3), 0.4)]
                for i, atom_value in enumerate(batch):
                    if self.model.logic_actor.atoms[i].pred.name == predicate.pred.name:
                        rule_atom.append((self.model.logic_actor.atoms[i], atom_value)) # Tuple of atom and its value. (on_ladder(x1,x2), 0.8)
                all_rule_state_atoms.append(rule_atom)
            
            all_rule_state_atoms.sort(key=lambda atom: len(atom[0][0].terms), reverse=True)

            self.grounded_rule_atoms = [] # Contains the atoms of all grounded rules. [[(on_ladder(x1,x2), 0.8), (same_level_ladder(x1,x2), 0.9)], [(on_ladder(x1,x3), 0.4), (same_level_ladder(x1,x3), 0.2)]]
            grounded_rule = [] # Serves as a list for the grounded rule in the combination algorithm, which will be appended to grounded_rule_atoms. E.g. [(on_ladder(x1,x2), 0.8), (same_level_ladder(x1,x2), 0.9)].
            current_obj = {} # Contains objects that have already been set in the combination algorithm. E.g. {P: obj1, D: obj2}
            self.all_object_pairs(all_rule_state_atoms, grounded_rule, current_obj, predicates_obj, predicates)

            product_grounded_rules = [] # Collects the product of the state atoms of all grounded rules. [0.8*0.9, 0.4*0.2]

            for grounded_rule in self.grounded_rule_atoms:
                product_atoms = 1
                for atom in grounded_rule:
                    product_atoms *= atom[1]
                product_grounded_rules.append(product_atoms)

            grounded_rules[clause.head] = self.grounded_rule_atoms[product_grounded_rules.index(max(product_grounded_rules))] # Add grounded rule that has highest value. E.g. {up_ladder: [(on_ladder(x1,x2), 0.8), (same_level_ladder(x1,x2), 0.9)]}

        index = -1.5

        predicate_indices, action_logic_prob, pred2prob_dict = self.selected_logic_rule()

        logic_policy_weight = self.model.actor.w_policy[1].tolist() # Determines how much influence the logic action probabilities have on the overall action probability distribution.

        rule_index = 0
        for rule, atoms in grounded_rules.items():
            
            index += 1.5
            rule_title = f"Clause: {rule.pred.name}"
            title = self.font.render(rule_title, True, "white", None)
            title_rect = title.get_rect()
            title_rect.topleft = (self.env_render_shape[0] + 10, 25 + index * 35)
            self.window.blit(title, title_rect)

            if rule_index in predicate_indices and self.model.actor.blender_mode != "neural" and action_logic_prob * logic_policy_weight > 0.1 and pred2prob_dict[rule_index] > 0.1:
                # Only highlight truth values of atoms from logic action rules that participated in the selection of the action with their assignment of a probability bigger than 0.1.
                # Another condition is a large enough weight for the logic policy during its blending with the neural module, as well as logic probability of the action.
                for atom in atoms:
                    index += 1
                    # Render cell background
                    color = atom[1] * CELL_BACKGROUND_HIGHLIGHT + (1 - atom[1]) * CELL_BACKGROUND_DEFAULT
                    pygame.draw.rect(self.window, color, [
                        anchor[0] - 2,
                        anchor[1] - 2 + index * 35,
                        (PREDICATE_PROBS_COL_WIDTH /2  - 12) * atom[1],
                        28
                    ])

                    text = self.font.render(str(f"{atom[1]:.3f} - {atom[0].pred.name}"), True, "white", None)
                    text_rect = text.get_rect()
                    text_rect.topleft = (self.env_render_shape[0] + 10, 25 + index * 35)
                    self.window.blit(text, text_rect)
            else:
                for atom in atoms:
                    index += 1
                    # Render cell background
                    text = self.font.render(str(f"{atom[0].pred.name}"), True, "white", None)
                    text_rect = text.get_rect()
                    text_rect.topleft = (self.env_render_shape[0] + 10, 25 + index * 35)
                    self.window.blit(text, text_rect)
            
            rule_index += 1
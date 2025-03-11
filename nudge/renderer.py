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
import saliency

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

        self.history = {'ins': [], 'obs': []} # For heat map
        

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

        self.heat_counter = -1

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

                self.heat_counter += 1

                self.update_history()

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
        #self._render_logic_rules() # Render the logic action rules and highlight the selected ones.
        #self._render_env()
        self._render_heat_map()

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

        predicate_indices = []
        action_logic_prob = 0

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
            pred2prob_dict = {} # Key is index of the predicate and value is the probability that it has assigned to the action.
            for j in range(len(indices.tolist()[0])):
                pred2prob_dict[indices.tolist()[0][j]] = predicate_probs[j]
            #print(action_predicates[action_indices[action]])
            
        logic_policy_weight = self.model.actor.w_policy[1].tolist() # Determines how much influence the logic action probabilities have on the overall action probability distribution.

        #print(self.model.actor.logic_action_probs[0].tolist())
        #print(logic_policy_weight)
        #print(action_logic_prob * logic_policy_weight)
        for i, rule in enumerate(logic_action_rules):
            is_selected = 0
            if i in predicate_indices and self.model.actor.actor_mode != "neural" and action_logic_prob * logic_policy_weight > 0.1 and pred2prob_dict[i] > 0.1:
                # Highlight predicates that contributed to the probability of the selected action with their assignment of a probability bigger than 0.1.
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


    def _render_heat_map(self, density=5, radius=5, prefix='default'):
        # Render normal game frame.
        if len(self.history['ins']) <= 1:
            self._render_env()

        # Render game frame with heat map.
        elif len(self.history['ins']) > 1:
            radius, density = 5, 5
            upscale_factor = 5
            actor_saliency = saliency.score_frame(
                self.env, self.model, self.history, self.heat_counter, radius, density, interp_func=saliency.occlude,
                mode="actor"
            )  # shape (84,84)

            critic_saliency = saliency.score_frame(
                self.env, self.model, self.history, self.heat_counter, radius, density, interp_func=saliency.occlude,
                mode="critic"
            )  # shape (84,84)

            frame = self.history['ins'][
                self.heat_counter].squeeze().copy()  # Get the latest frame with shape (210,160,3)

            frame = saliency.saliency_on_atari_frame(actor_saliency, frame, fudge_factor=400, channel=2)
            frame = saliency.saliency_on_atari_frame(critic_saliency, frame, fudge_factor=600, channel=0)

            frame = frame.swapaxes(0, 1).repeat(upscale_factor, axis=0).repeat(upscale_factor, axis=1)  # frame has shape (210,160,3), upscale to (800,1050,3). From ocatari/core.py/render()

            heat_surface = pygame.Surface(self.env_render_shape)

            pygame.pixelcopy.array_to_surface(heat_surface, frame)

            self.window.blit(heat_surface, (0, 0))

    def update_history(self):
        raw_state, _, _, _, _ = self.env.env.step(self.action) # raw_state has shape (4,84,84), from step() in env.

        # save info!
        self.history['ins'].append(self.env.env._env.og_obs)  # Original rgb observation with shape (210,160,3)
        self.history['obs'].append(raw_state)  # shape (4,84,84), no prepro necessary
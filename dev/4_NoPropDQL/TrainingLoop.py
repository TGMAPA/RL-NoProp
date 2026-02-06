import random, torch
import torch.nn.functional as F


# Bandits-DQN Training Loop
def train(env,
          episodes,
          mlps,
          optimizers,
          diffuser,
          epsilon,
          epsilon_min,
          epsilon_decay,
          ):
    
    q_history = []
    
    # Training Loop
    for i in range(1, episodes+1):

        # Select action with epsilon-greedy
        if random.random() < epsilon:
            # Select random action: Exploration
            action = random.randrange(env.num_actions)
        else:
            # Select action with highest Q value
            with torch.no_grad():
                # Get noisy vector 
                z = torch.randn(1, env.num_actions)

                # Inverse Diffusion (Inference)
                for t in reversed(range(diffuser.T)):
                    z = mlps[t](torch.tensor(env.state).unsqueeze(0), z)

                # Get the action with the highest Q-value (SLot 1 or 2)
                action = torch.argmax(z).item()

        # Execute action
        reward = env.step(action)

        # Q target
        q_target = torch.zeros(1, env.num_actions)
        q_target[0, action] = reward

        # DIffuse q target (Direct Diffusion)
        z = diffuser.diffuse(q_target)

        # NoProp Training
        for t in range(diffuser.T):
            pred = mlps[t](torch.tensor(env.state).unsqueeze(0), z[t+1].detach())
            loss = ((pred - q_target) ** 2).mean()

            optimizers[t].zero_grad()
            loss.backward()
            optimizers[t].step()

            # Logging
            if i% 50 == 0:
                with torch.no_grad():
                    # Get noisy vector 
                    z_inference = torch.randn(1, env.num_actions)

                    # Inverse Disffusion (Inference)
                    for t in reversed(range(diffuser.T)):
                        z_inference = mlps[t](torch.tensor(env.state).unsqueeze(0), z_inference)
                        
                    q_history.append(z_inference.squeeze(0).cpu().numpy())

        # Update epsilon
        epsilon = max(epsilon_min, epsilon * epsilon_decay)

    return q_history
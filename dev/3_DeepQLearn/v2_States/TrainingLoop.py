import random, torch
import torch.nn.functional as F


# Bandits-DQN Training Loop
def train(env,
          onlineNet, 
          targetNet,
          buffer,
          episodes,
          optimizer,
          epsilon,
          epsilon_min,
          epsilon_decay,
          batch_size = 32,
          targetNet_update = 100
          ):
    
    q_history = []
    
    # Training Loop
    for i in range(1, episodes+1):
        # Get random state ( [1,0,0,0, 0,1,0,0] )
        state = env.getState() 

        # Select action with epsilon-greedy
        if random.random() < epsilon:
            # Select random action: Exploration
            action = random.randrange(env.num_actions)
        else:
            # Select action with highest Q value
            with torch.no_grad():
                q_values = onlineNet(state.unsqueeze(0))
                action = torch.argmax(q_values).item()

        # Execute action
        reward = env.step(action)

        # Store experience in buffer
        buffer.push( (state, action, reward) )


        # Update net's weights when buffer is long enough
        if len(buffer) >= batch_size:
            # Get a random batch from buffer
            batch = buffer.sample(batch_size)
            
            # Unpack batch
            states, actions, rewards = zip(*batch)
            states  = torch.stack(states)
            actions = torch.tensor(actions, dtype=torch.int64).unsqueeze(1)
            rewards = torch.tensor(rewards, dtype=torch.float32).unsqueeze(1)

            # Get q_value through net
            q_values = onlineNet(states).gather(1, actions)

            #Bandit target
            targets = rewards

            # Loss
            loss = F.mse_loss(q_values, targets)

            # Optimization
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Log Q values for analysis
            if i% 10 == 0:
                with torch.no_grad():
                    q = onlineNet(
                        state.unsqueeze(0)
                    )
                    q_history.append(q.squeeze(0).cpu().numpy())


        # Update target network
        if i % targetNet_update == 0:
            targetNet.load_state_dict(onlineNet.state_dict())

        # Update epsilon
        epsilon = max(epsilon_min, epsilon * epsilon_decay)

    return q_history
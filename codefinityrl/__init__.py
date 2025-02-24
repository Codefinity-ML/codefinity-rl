from gymnasium import register

register(
  id='MultiArmedBanditStationary-v0',
  entry_point='codefinityrl.envs:MultiArmedBanditStationaryEnv',
)

register(
  id='MultiArmedBanditDynamic-v0',
  entry_point='codefinityrl.envs:MultiArmedBanditDynamicEnv',
)
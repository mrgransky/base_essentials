# Q1)
# Input: nums = [2, 7, 11, 15], target = 9
# Output: [0, 1]
# Explanation: nums[0] + nums[1] = 2 + 7 = 9

# nums = [2, 7, 11, 15]
nums = [5, 12, -6, 2, 7, 11, 15, 12, 2]
target = 9


indices = list()
for idx1, val1 in enumerate(nums):
	# print(idx1, val1)
	for idx2, val2 in enumerate(nums[idx1+1:], start=idx1+1):
		# print(idx2, val2)
		if val1 + val2 == target:
			indices.append((idx1, idx2))
			# print(f"success: {idx1}, {idx2}")
			break

# print(indices)
# [(2, 6), (3, 4), (4, 8)]


# Q2)
# Given a string s, find the length of the longest substring without repeating characters.
# s = "pwwkew"
for s in ["abcabcbb", "bbbbb", "pwwkew", "dvdf", "abccdef", "abba",""]:
	seen = {}
	duplicate_idx = 0
	duplicate_char=None
	best = 0
	print(s)
	for idx, ch in enumerate(s):
		print(f"idx={idx}")
		print(f"seen={seen} duplicate: (idx={duplicate_idx}, {duplicate_char})")
		print(f"ch={ch}")
		if ch in seen and seen[ch] >= duplicate_idx:
			print(f"XX duplicate '{ch}' XX move duplicate_idx {duplicate_idx} -> {seen[ch]+1}")
			duplicate_idx = seen[ch] + 1
			duplicate_char = ch
		seen[ch] = idx
		current_window = s[duplicate_idx:idx+1]
		best = max(best, idx-duplicate_idx+1)
		print(f"current_window='{current_window}' seen={seen} (duplicate: idx={duplicate_idx}, {duplicate_char}) best={best}")
		print()
	print("-"*100)

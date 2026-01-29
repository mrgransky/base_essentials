#!/bin/bash
set -e  # Exit on error

echo "================================================"
echo "NLTK Manual Download (IPv4 forced)"
echo "================================================"

# Create directories
echo -e "\n[1/8] Creating directories..."
mkdir -p ~/nltk_data/corpora
mkdir -p ~/nltk_data/tokenizers
mkdir -p ~/nltk_data/taggers
echo "✅ Directories created"

# Function to download with fallback
download_file() {
		local url=$1
		local output=$2
		local name=$3
		
		echo -e "\n[Downloading] $name..."
		
		# Try curl first (usually works better)
		if command -v curl &> /dev/null; then
				echo "  Using curl..."
				if curl -4 -L -f -o "$output" "$url" --connect-timeout 30 --max-time 300; then
						echo "  ✅ Downloaded with curl"
						return 0
				else
						echo "  ❌ curl failed"
				fi
		fi
		
		# Try wget with IPv4
		if command -v wget &> /dev/null; then
				echo "  Using wget (IPv4 only)..."
				if wget -4 -O "$output" "$url" --timeout=30 --tries=3; then
						echo "  ✅ Downloaded with wget"
						return 0
				else
						echo "  ❌ wget failed"
				fi
		fi
		
		echo "  ❌ Failed to download $name"
		return 1
}

# Download punkt
echo -e "\n[2/8] Downloading punkt..."
cd ~/nltk_data/tokenizers
if download_file \
		"https://raw.githubusercontent.com/nltk/nltk_data/gh-pages/packages/tokenizers/punkt.zip" \
		"punkt.zip" \
		"punkt"; then
		unzip -q punkt.zip && rm punkt.zip
		echo "✅ punkt installed"
else
		echo "⚠️ punkt download failed"
fi

# Download punkt_tab
echo -e "\n[3/8] Downloading punkt_tab..."
if download_file \
		"https://raw.githubusercontent.com/nltk/nltk_data/gh-pages/packages/tokenizers/punkt_tab.zip" \
		"punkt_tab.zip" \
		"punkt_tab"; then
		unzip -q punkt_tab.zip && rm punkt_tab.zip
		echo "✅ punkt_tab installed"
else
		echo "⚠️ punkt_tab download failed"
fi

# Download stopwords
echo -e "\n[4/8] Downloading stopwords..."
cd ~/nltk_data/corpora
if download_file \
		"https://raw.githubusercontent.com/nltk/nltk_data/gh-pages/packages/corpora/stopwords.zip" \
		"stopwords.zip" \
		"stopwords"; then
		unzip -q stopwords.zip && rm stopwords.zip
		echo "✅ stopwords installed"
else
		echo "⚠️ stopwords download failed"
fi

# Download wordnet
echo -e "\n[5/8] Downloading wordnet..."
if download_file \
		"https://raw.githubusercontent.com/nltk/nltk_data/gh-pages/packages/corpora/wordnet.zip" \
		"wordnet.zip" \
		"wordnet"; then
		unzip -q wordnet.zip && rm wordnet.zip
		echo "✅ wordnet installed"
else
		echo "⚠️ wordnet download failed"
fi

# Download omw-1.4
echo -e "\n[6/8] Downloading omw-1.4..."
if download_file \
		"https://raw.githubusercontent.com/nltk/nltk_data/gh-pages/packages/corpora/omw-1.4.zip" \
		"omw-1.4.zip" \
		"omw-1.4"; then
		unzip -q omw-1.4.zip && rm omw-1.4.zip
		echo "✅ omw-1.4 installed"
else
		echo "⚠️ omw-1.4 download failed"
fi

# Download averaged_perceptron_tagger (old version - for compatibility)
echo -e "\n[7/8] Downloading averaged_perceptron_tagger..."
cd ~/nltk_data/taggers
if download_file \
		"https://raw.githubusercontent.com/nltk/nltk_data/gh-pages/packages/taggers/averaged_perceptron_tagger.zip" \
		"averaged_perceptron_tagger.zip" \
		"averaged_perceptron_tagger"; then
		unzip -q averaged_perceptron_tagger.zip && rm averaged_perceptron_tagger.zip
		echo "✅ averaged_perceptron_tagger installed"
else
		echo "⚠️ averaged_perceptron_tagger download failed"
fi

# Download averaged_perceptron_tagger_eng (NEW - required for NLTK 3.9+)
echo -e "\n[8/8] Downloading averaged_perceptron_tagger_eng..."
if download_file \
		"https://raw.githubusercontent.com/nltk/nltk_data/gh-pages/packages/taggers/averaged_perceptron_tagger_eng.zip" \
		"averaged_perceptron_tagger_eng.zip" \
		"averaged_perceptron_tagger_eng"; then
		unzip -q averaged_perceptron_tagger_eng.zip && rm averaged_perceptron_tagger_eng.zip
		echo "✅ averaged_perceptron_tagger_eng installed"
else
		echo "⚠️ averaged_perceptron_tagger_eng download failed"
fi

# Verification
echo -e "\n================================================"
echo "VERIFICATION"
echo "================================================"

echo -e "\nChecking installed packages:"
for dir in ~/nltk_data/*/; do
		echo "  📁 $dir"
		ls -1 "$dir" | sed 's/^/    - /'
done

echo -e "\n✅ Installation complete!"
echo -e "\nTest in Python with:"
echo "  python -c \"import nltk; print(nltk.word_tokenize('Test sentence.'))\""
echo "  python -c \"import nltk; print(nltk.pos_tag(['This', 'is', 'a', 'test']))\""
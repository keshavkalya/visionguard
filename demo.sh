# Quick demo of the VisionGuard API using curl.
# Start the server first in another terminal: python app.py
# Usage: sh demo.sh path/to/your/photo.jpg

PHOTO=$1

if [ -z "$PHOTO" ]; then
  echo "Usage: sh demo.sh path/to/your/photo.jpg"
  exit 1
fi

echo "--- 1. Health check (who does the server know?) ---"
curl -s http://localhost:5001/health
echo
echo

echo "--- 2. Enroll yourself as 'demo_user' with your photo ---"
curl -s -X POST http://localhost:5001/enroll -F "name=demo_user" -F "image=@$PHOTO"
echo
echo

echo "--- 3. Verify the same photo (should recognize you now) ---"
curl -s -X POST http://localhost:5001/verify -F "image=@$PHOTO"
echo
echo

echo "Done! To remove the demo user again: delete dataset/demo_user/ and re-run train.py"

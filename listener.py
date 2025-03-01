import speech_recognition as sr

class ContinuousSpeechToText:
    def __init__(self):
        self.recognizer = sr.Recognizer()
        self.microphone = sr.Microphone()

    def listen_and_convert(self):
        """
        Continuously listens to microphone input and converts detected speech to text.
        :return: Recognized text
        """
        print("The system is continuously listening. You can speak at any time. Press Ctrl+C to stop the program.")

        try:
            with self.microphone as source:
                self.recognizer.adjust_for_ambient_noise(source)  # Automatically adjust for ambient noise
                print("Ambient noise adjustment completed. Start listening...")

                while True:
                    print("Waiting for speech input...")
                    # Continuously listen until speech is detected
                    audio = self.recognizer.listen(source)
                    print("Speech input detected, starting recognition...")

                    try:
                        # Use Google's Speech Recognition API to convert speech to text
                        text = self.recognizer.recognize_google(audio, language="en-US")
                        print(f"Recognition result: {text}")
                        return text
                    except sr.UnknownValueError:
                        print("Could not understand the speech. Please try again.")
                    except sr.RequestError as e:
                        print(f"Speech recognition service error: {e}")
                    except Exception as e:
                        print(f"An error occurred: {e}")

        except KeyboardInterrupt:
            print("\nProgram terminated.")
        except Exception as e:
            print(f"An error occurred: {e}")

# Test code (executed only when running this file directly)
if __name__ == "__main__":
    stt = ContinuousSpeechToText()
    stt.listen_and_convert()

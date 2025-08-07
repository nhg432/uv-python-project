from uv import UV

def main():
    app = UV()

    @app.route('/')
    def home():
        return "Welcome to the UV Python Project!"

    app.run()

if __name__ == "__main__":
    main()
package socket;

import assign.Game;
import gui.FrameControl;


import java.io.IOException;
import java.net.ServerSocket;
import java.net.Socket;


//game using sockets
public class GameServer {

    @SuppressWarnings("resource")
    public static void main(String[] args) throws IOException {
        Game game = new Game();
        ServerSocket server = new ServerSocket(SocketConstant.PORT);
        System.out.println("Started GameService at port "+ SocketConstant.PORT);
        System.out.println("Connecting....");
        FrameControl frame = new FrameControl("The Stock Game", game);
        frame.setVisible(true);

        //maintain connection
        while(true){

            Socket socket = server.accept();
            System.out.println("Connect!");
            GameService service = new GameService(game, socket);

            new Thread(service).start();

        }
    }
}

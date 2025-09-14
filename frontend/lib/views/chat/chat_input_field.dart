import 'package:flutter/material.dart';

class ChatInputField extends StatefulWidget {
  final VoidCallback onSendPressed;
  const ChatInputField({super.key, required this.onSendPressed});

  @override
  State<ChatInputField> createState() => _ChatInputFieldState();
}

class _ChatInputFieldState extends State<ChatInputField> {
  final TextEditingController _controller = TextEditingController();

  @override
  Widget build(BuildContext context) {
    return Row(
      children: [
        Expanded(
          child: TextField(controller: _controller),
        ),
        IconButton(
          icon: const Icon(Icons.send),
          onPressed: widget.onSendPressed,
        ),
      ],
    );
  }
}

import 'package:flutter/material.dart';

import 'json_code_snippet_view.dart';

class AgentMessageTile extends StatefulWidget {
  final String message;
  const AgentMessageTile({super.key, required this.message});

  @override
  State<AgentMessageTile> createState() => _AgentMessageTileState();
}

class _AgentMessageTileState extends State<AgentMessageTile> {
  bool expanded = false;

  @override
  Widget build(BuildContext context) {
    return Column(
      crossAxisAlignment: CrossAxisAlignment.start,
      children: [
        const Text('Agent'),
        Text(widget.message),
        IconButton(
          icon: Icon(expanded ? Icons.keyboard_arrow_up : Icons.keyboard_arrow_down),
          onPressed: () {
            setState(() {
              expanded = !expanded;
            });
          },
        ),
        if (expanded)
          const JsonCodeSnippetView(jsonString: '{}'),
      ],
    );
  }
}
